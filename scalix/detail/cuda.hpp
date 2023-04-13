#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace sclx::cuda {

class task_scheduler;

namespace detail {

struct virtual_task {
    virtual void operator()() = 0;
    virtual ~virtual_task()   = default;
};

template<class ReturnType, class... Args>
struct packaged_task final : virtual_task {
    std::function<ReturnType(Args...)> function;
    std::promise<ReturnType> promise;

    explicit packaged_task(std::function<ReturnType(Args...)> function)
        : function(std::move(function)) {}

    void operator()() override { promise.set_value(function()); }
};

class cuda_thread_pool {
  public:
    cuda_thread_pool()                                   = delete;
    cuda_thread_pool(const cuda_thread_pool&)            = delete;
    cuda_thread_pool(cuda_thread_pool&&)                 = delete;
    cuda_thread_pool& operator=(const cuda_thread_pool&) = delete;
    cuda_thread_pool& operator=(cuda_thread_pool&&)      = delete;
    friend class ::sclx::cuda::task_scheduler;

    ~cuda_thread_pool() {
        stop = true;
        for (auto& thread : threads) {
            thread.join();
        }
    }

  private:
    std::vector<std::queue<std::unique_ptr<virtual_task>>> queues;
    std::vector<std::thread> threads;
    std::vector<std::mutex> queue_mutexes;
    std::atomic_bool stop = false;

    explicit cuda_thread_pool(int num_threads)
        : queues(num_threads),
          queue_mutexes(num_threads) {
        for (int i = 0; i < num_threads; i++) {
            init_thread(i);
        }
    }

    void init_thread(int thread) {
        auto thread_loop = [this, thread]() {
#ifdef SCALIX_EMULATE_MULTIDEVICE
            cudaSetDevice(0);
#else
            cudaSetDevice(thread);
#endif
            while (true) {
                std::unique_ptr<virtual_task> task;
                {
                    std::lock_guard<std::mutex> lock(queue_mutexes[thread]);
                    if (stop && queues[thread].empty()) {
                        break;
                    }
                    if (!queues[thread].empty()) {
                        task = std::move(queues[thread].front());
                        queues[thread].pop();
                    }
                }
                if (task) {
                    (*task)();
                } else {
                    std::this_thread::yield();
                }
            }
        };
        threads.emplace_back(thread_loop);
    }

    template<class ReturnType, class... Args>
    std::future<ReturnType>
    submit_task(int thread, std::function<ReturnType(Args...)> function) {
        std::lock_guard<std::mutex> lock(queue_mutexes[thread]);
        auto task_
            = std::make_unique<packaged_task<ReturnType, Args...>>(function);
        auto future = task_->promise.get_future();
        queues[thread].push(std::move(task_));
        return future;
    }
};
}  // namespace detail
}  // namespace sclx::cuda
