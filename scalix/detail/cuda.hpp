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

struct packaged_task_interface {
    virtual void operator()()             = 0;
    virtual ~packaged_task_interface() = default;
};

template<class R>
struct packaged_task_wrapper : public packaged_task_interface {
    std::packaged_task<R()> task;

    explicit packaged_task_wrapper(std::packaged_task<R()>&& task)
        : task(std::move(task)) {}

    void operator()() override { task(); }
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
    std::vector<std::queue<std::unique_ptr<packaged_task_interface>>> queues;
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
                std::unique_ptr<packaged_task_interface> task;
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

    template<class F, class... Args>
    std::future<std::invoke_result_t<F, Args...>>
    submit_task(int device_id, F&& f, Args&&... args) {

        auto args_tuple = std::make_tuple(args...);

        auto task_lambda = [f = std::forward<F>(f), args_tuple]() mutable {
            return std::apply(f, args_tuple);
        };

        using return_type = std::invoke_result_t<F, Args...>;
        auto task         = std::packaged_task<return_type()>(task_lambda);
        auto future       = task.get_future();
        std::lock_guard<std::mutex> lock(queue_mutexes[device_id]);
        queues[device_id].emplace(
            new packaged_task_wrapper<return_type>(std::move(task))
        );
        return future;
    }
};
}  // namespace detail
}  // namespace sclx::cuda
