# Release 0.5.0

This will likely be one of, if not the last release of pre-1.0 Scalix (see
`Roadmap.md` for more information). Up to this point, I've been adding features
and fixing bugs as I've needed them for my own research. However, that project
is coming to a close so Scalix features will likely be paused until the paper is
submitted for publication.

With the project I've been working, a highly parallel fluid simulation method
for acoustics, the initial hope was to have a distributed implementation across
many GPUs, hence Scalix was born. Unfortunately, the reliance of Scalix on CUDAs
unified memory management ended up being a bottleneck for the project, so the
distributed implementation was never used, despite having automatic support via
Scalix's opaque distributed API, because one particular part of the method was
not conducive to CUDAs managed memory heuristics.

Using Scalix extensively for this project has been incredibly useful, but the
pre-1.0 APIs are clunky, requiring complicated workarounds for use cases that
weren't considered upon initial design. Rather than continuing pre-1.0
development, I will begin development of the 1.0 version, which will use SYCL as
a base. Using what I've learned from my usage so far, I'm much more confident in
the design of the 1.0 version, and I'm excited to get started. Some proof of
concepts have already been explored and the performance is looking promising.

Without further ado, I present tha changes for the final pre-1.0 release of
Scalix, except for possible bug fixes. Next major entry in this log will be a
beta release of 1.0!

## Features

- Serialization support for `sclx::array` types using `cereal`
- STL-like methods:
  - `sclx::algorithm::count_if`
  - `sclx::algorithm::min_element`/`max_element`
  - `sclx::algorithm::elementwise_reduce` now supports concurrent execution'
  - `sclx::iota`
- Kernel info now provides information about the grid stride loop, useful for
  prefetching
- better `sclx::array` constructors, avoiding macro guards for host vs device
  implementations
- Better implicit casting of `sclx::array` non-const types to const types

## API Changes

- Lots of changes of pass-by-reference to pass-by-value for `sclx::array` types.
  Not only is the cost of copying shared pointers minimal compared to the
  typical computational cost of various numerical algorithms, but passing by
  value improves thread safety.
- `sclx::cexpr_memcpy` is now `sclx::constexpr_assign_array`, better aligning
  with what the function actually does

## Bug Fixes

- `sclx::local_array` now aligns allocations to its respective type's alignment,
  alleviating a rare bug that caused CUDA kernels to fail with a cryptic error
  message

# Release 0.4.1

Minor release, fixed version in `CMakelists.txt`

# Release 0.4.0

## Features

- Kernel launches now throw when requesting too much shared memory
- More CUDA API error checks
- Added transformation algorithm for `sclx::array` types
- Added element-wise reduction algorithm for `sclx::array` types. Algorithm
  accepts arbitrary number of inputs greater than or equal to 2, allowing for
  reduction of many arrays in one kernel launch.
- Can assign one array to another via `sclx::assign_from`
- When emulating multiple devices, the memory query divides available and total
  memory by the number of emulated devices.
- Improved array constructors
- Can now copy raw pointer data to sub-ranges of arrays
- Added custom `std::swap` specialization for `sclx::array` types for possible
  performance improvements
- Can now construct `sclx::array_list` from pointer to arrays
- CUDA Unified Memory hints are now disabled if only one device is detected,
  leading to large performance improvements. Note that this means problems will
  scale poorly from 1 to 2 devices, due to the performance hit induced by
  enabling Unified Memory hints. Scaling with >2 devices should be fine. Unified
  Memory hints can be re-enabled via `SCLX_DISABLE_SINGLE_DEVICE_OPTIMIZATION`
  preprocessor definition.
- If a user requests more shared memory than the default `48KiB`, then the
  runtime will attempt to request more shared memory for the relevant kernel,
  and will throw if it can't.

## API Changes

- Added backwards-compatible default template parameters for `sclx::kernel_info`
  allowing the user to provide `const sclx::kernel_info<> &info` to kernels with
  a thread block rank of `1` and input range of rank `1`, the most common use
  case.

## Bug Fixes

- Fixed incorrect capture order in memory query APIs that caused `free` and
  `total` to be swapped
- Fixed a bugs in macro used to define custom kernel tags
- Fixed bug in `sclx::array::set_primary_devices` that didn't propagate the
  prefetch flag down the call stack
- Fixed issue with kernel launches where the `sclx::kernel_handler` is not
  passed as a const reference causing compiler errors
- Fixed a bug in `sclx::local_array` caused by incorrectly marked const methods

# Release 0.3.1

Forgot to run autoformat script before `0.3.0` release. This release is the
same, but with the autoformat script run. For more information, see the `0.3.0`
[release notes](https://github.com/NAGAGroup/Scalix/releases/tag/0.3.0)

# Release 0.3.0

## Features

- Added some useful operator overloading for `sclx::md_index_t`
- Kernels are now provided with a `sclx::kernel_info` object which provides
  information about the current kernel execution such as global and local thread
  id's, global vs device range sizes, etc.
- A reduction algorithm for reducing along the last dimension of a
  multidimensional array, `sclx::algorithm::reduce_last_dim`. The resulting
  array has one less dimension than the original array.
- Better exceptions for identifying where errors occur in the library
- `sclx::shape_like_t` derivatives can now be constructed from C-style arrays,
  useful for something like `sclx::shape_t<2>({100, 100})`
- Added `sclx::zeros` and `sclx::ones` functions for creating arrays of zeros
  and ones in `scalix/fill.cuh`
- Added memory query APIs for both host and devices
- Added a `sclx::cuda::task_scheduler` that guarantees a task is executed on a
  specific device in its own thread, useful when calling to external libraries
  like `cuBLAS`

## API Changes

- `sclx::md_index_t` now has better method names for converting to/from linear
  indices
- Returned data pointer for `sclx::array::data()` is no longer constant by
  default.

## Bug Fixes

- Fixed memory leak issue in `sclx::detail::unified_ptr` that only occurred for
  exotic use cases
- Fixed issues with some APIs not working when using `sclx::array` data type was
  const-qualified
- Some bug fixes to the example programs

---

# Release 0.2.0

This release is a result of the work on the `rectangular_partitioner` in the
`NAGA` library (not yet public as of this writing). The key takeaway from this
release is the new "index generator"-based CUDA kernels, expanding the type of
distributed workloads that can be executed on Scalix.

## Features

- `inclusive_scan`
- `fill`
- Wrapped `std::filesystem` namespace in `sclx::filesystem`, allowing for our
  extensions to the standard library and the standard library itself to exist
  side-by-side in the same namespace.
- Implemented an "index generator"-based CUDA kernel. Useful for dynamic,
  unstructured write patterns which cannot be distributed using the typical
  range-based kernel. Instead, the workload is repeated across all devices, but
  the index provided by the index generator determines if the provided kernel
  functor is actually called, ensuring writes are localized to the correct
  device.
- Added pointer cast implementations for `sclx::detail::unified_ptr`
- Allow for unevenly distributed device splits using
  `sclx::array::set_primary_devices`

## API Changes

- Memory advice hints in `cuda::traits` are now `enum` instead of `struct` to
  allow for parameter type to APIs
- Conversion APIs between multidimensional and linear indices have new names to
  better reflect their purpose
- `sclx::data_capture_mode::shared` has been renamed to
  `sclx::data_capture_mode::capture` to better reflect its purpose
