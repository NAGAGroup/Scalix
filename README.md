# Welcome to SCALIX!
---
**Note**: This proof-of-concept version of Scalix has reached EOL and will no longer receive
updates/fixes, except potentially on an as-needed basis for current research that depends on this
version. The insights gained since starting this project, both through active development and
real-world use in a non-trivial codebase, have been incredibly invaluable. With those insights,
development on a much improved version written in SYCL instead of CUDA has begun on the sycl
branch. This version will be better in nearly every way, some of the improvements include:

- SYCL is a more modern API, using ISO compliant C++ instead of an extended C-like language with
  some C++ support like CUDA. Not only does this enable the nice-to-haves of modern C++, but the
  new implementation will be platform agnostic, enabling a wider range of hardware, and likely
  even more added hardware support in the future with no added development of Scalix
- Using the SYCL specification as a guide, Scalix will use the buffer/accessor model, rather
  than relying on CUDA's Unified Memory(UM)/On-demand page migration. This will allow the Scalix
  runtime to maintain a dependency graph and explicitly manage data in an optimal way defined by
  that dependency graph. While Scalix currently scales well with relatively simple access patterns,
  the UM driver heuristics really break down for more complex problems leading to inefficient data
  transfers that tank performance
- Way simpler API that is both easier to use and to maintain/test. If you know SYCL, you'll feel right
  at home with Scalix
- Much better thread management. Currently, Scalix spawns three or four nested threads for a single kernel, which
  is already pretty bad, but the threads are spawned via `std::async` instead of a dedicated threadpool. While the
  true cost of this is unclear, it is a pretty bad implementation that has a lot of room for improvement and makes
  profiling incredibly difficult in NSight as the number of CUDA streams balloons very quickly. Going forward,
  Scalix will internally use a multi-threading approach that is akin to an actor model and we have been able to
  avoid spawning any threads until absolutely necessary, completely avoiding any need for async tasks to spin in
  some thread waiting on a set of dependent futures before proceeeding. It also enables a stateless implementation
  where the dependency graph is implicitly defined via shared/unique locks around protected data.
  Additionally, we will likely be incorporating a threadpool for even better thread management.
- Well-thought out API/system design and proper unit-testing, static analysis, etc from the start. With the first
  attempt at Scalix, it was being developed in parallel with a research project, and often times getting something
  that worked, even if not ideal, was priortized over well-thought out solutions. Well, that version of Scalix works
  well enough for the research project, at least until the first working release of SYCL-Scalix, so we can take
  our time. This should give more confidence to future users that the project is properly handled, and will make
  getting started with the software much easier.
- More opportunities for optimization, some implicit, some explicit:
    - We will enforce that only the minimally-required memory to keep data valid will be used unless explicitly declared by the user.
    - The way memory is handled will have very clear, documented behavior, allowing users to fine-tune their code for these rules.
    - Only the memory locations being accessed on each device will be allocated, where these accesses are specified by either a few
      simple but commonly used strategies or a more flexible strategy where the user provides a custom command that defines accessed indices.
    - In terms of optimization of compute resources, Scalix development will be focused on providing default and user-defined "knobs"
      that can be tweaked to find the optimal balance of resources
---

Scalix is a data parallel computing framework designed to provide an easy-to-use
interface for designing data parallel applications that automatically scale to
the available compute resources. Data distribution and load balancing are
abstracted away from the user, allowing them to focus on the application logic.
Scalix is designed to be written using "kernel style" programming, where the
user writes a single function that is executed on a grid of threads.

Currently, Scalix is only supported on a single machine with multiple CUDA
devices that must support Unified Memory. Furthermore, while not strictly
required, it is recommended to use Linux and a CUDA device with compute
capability 6.0 or higher so that we can take advantage of on-demand page
migration. Performance without on-demand page migration is not tested, but will
assuredly be slower.

In the future, we plan to migrate to a SYCL-based implementation + MPI. This
future specification will look drastically different as we won't be able to rely
on Unified Memory, but many of the concepts will remain the same. Specifically,
the user should be able to write for an abstract "device" that may be one or
many actual compute devices.

Even further in the future we plan to define a generic specification for which
anyone can write their own implementation.

Think of this current version of Scalix as a proof of concept.

For a more detailed outline of our roadmap, see [ROADMAP.md](ROADMAP.md).

## Getting Started

For the moment, this is a header-only library, but that may change in the
future. Any program that uses Scalix must be compiled with the nvcc flag
`--extended-lambda`. The library is written in C++17. We have provided a
`CMakeLists.txt` file that provides the `scalix` interface library that, when
added as a link to a target, will automatically add the `--extended-lambda` flag
to the nvcc compiler. There are also a number of options that can be used to do
things like emulate multiple devices to check for correctness. These options are
described in the `CMakeLists.txt` file.

### Writing a Scalix Program

In the following program we add one to all elements of a Scalix array.

```c++
#include <scalix/scalix.cuh>

int main() {
    // create a 2D array of 1000x1000 elements
    //
    // the unified memory hints for the array are set such as to maximize
    // performance for equally distributed access across all devices
    sclx::array<int, 2> arr({1000, 1000});

    // initialize the array to 0
    //
    // kernels automatically distribute the work across all devices
    // for the result array
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<2>(arr.shape()),
            arr,  // need to provide a result array to guide the distribution of
                  // work
            [=] __device__(const sclx::md_index_t<2>& idx,
                           /*see below for what this is*/ const auto&) {
                arr[idx] = 0;
            }
        );
    });

    // we also provide a kernel info object (sclx::kernel_info<RangeRank,
    // ThreadBlockRank>) add one to all elements and print out the block thread
    // id
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<2>(arr.shape()),
            arr,
            [=] __device__(
                const sclx::md_index_t<2>& idx,
                const sclx::kernel_info<
                    2,
                    sclx::cuda::traits::kernel::default_block_shape.rank()>&
                    info
            ) {
                printf(
                    "block thread id: %u\n",
                    static_cast<uint>(info.local_thread_id()[0])
                );
                arr[idx] += 1;
            }
        );
    });
}
```

A special type of problem one may face is one where the write pattern cannot fit
into a neat range. For example, suppose we want to generate a histogram for some
values in an array. Sequentially, we would do something like iterate over each
value, determine the bin it belongs to, and increment the value of that bin.

To do this in Scalix, we instead would use the "index generator"-based kernel
API. The index generator in this case would take an index from the range
associated with the values and return the bin index. Our kernel implementation
replicates the entire range over each device, but only calls the provided
functor if the write index is local to the device. So, unfortunately, this means
scaling will not be ideal, and in many cases worse than a single device
implementation. However, what this does allow is the prevention of expensive
memory transfers for a problem that mostly scales well using the range-based
API.

```c++
#include <scalix/scalix.cuh>

// required interface for an index generator
class histogram_index_generator {
  public:
    static constexpr uint range_rank = 1;
    static constexpr uint index_rank = 1;

    __host__ __device__ const sclx::md_range_t<range_rank>& range() const;

    __host__ __device__ const sclx::md_range_t<index_rank>& index_range() const;

    __host__ __device__ sclx::md_index_t<index_rank>
    operator()(const sclx::md_index_t<range_rank>&) const;

    ...
};

int main() {

    sclx::array<int, 1> values({1000});
    sclx::array<int, 1> histogram({100});

    histogram_index_generator index_generator(...);

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            index_generator,
            histogram,
            [=] __device__(const sclx::md_index_t<1>& histogram_idx, const auto&) {
                atomicAdd(&histogram[histogram_idx], 1);
            }
        );
    });
}

```

The [examples](examples) directory contains a number of examples that
demonstrate how to use the library. For every new feature we add, we will do our
best to add an example that demonstrates its use. As we approach the first
stable release, we will start to focus on adding proper documentation.

## Performance

We have shown almost perfect strong scaling for the `distributed_access` example
in the repo, distributed across 2 GPUs. The example has two arrays of interest.
Each kernel reads 64 elements from one array and atomically adds it to an
element of the other array for each element in the result array. A portion of
these reads happen across device boundaries. Also note that the source and
result arrays exchange roles for each kernel, showing we can still get good
performance, even for arrays that read/write equally. Finally, a distributed
reduction algorithm is used to sum the values of one of the arrays together just
for fun (and to check that the solution is correct). The following times were
recorded on 1 vs 2 RTX 3090s, respectively.

```
1 GPU time: 163.1ms
2 GPU time: 87.5ms
```
