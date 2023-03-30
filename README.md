# Welcome to SCALIX!

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
                arr,  // need to provide a result array to guide the distribution of work
                [=] __device__ (const sclx::index_t<2> &idx) {
                    arr[idx] = 0;
                }
        );
    });

    // add one to all elements
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
                sclx::md_range_t<2>(arr.shape()),
                arr,
                [=] __device__ (const sclx::index_t<2> &idx) {
                    arr[idx] += 1;
                }
        );
    });
}
```

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
