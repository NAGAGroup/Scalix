# Release 0.3.0

## Features

- Added some useful operator overloading for `sclx::md_index_t`
- Kernels are now provided with a `sclx::kernel_info` object which provides
  information about the current kernel execution such as global and local thread id's, global vs device range sizes, etc.
- A reduction algorithm for reducing along the last dimension of a multidimensional array, `sclx::algorithm::reduce_last_dim`. The resulting array has one less dimension than the original array.
- Better exceptions for identifying where errors occur in the library
- `sclx::shape_like_t` derivatives can now be constructed from C-style arrays, useful for something like `sclx::shape_t<2>({100, 100})`
- Added `sclx::zeros` and `sclx::ones` functions for creating arrays of zeros and ones in `scalix/fill.cuh`
- Added memory query APIs for both host and devices
- Added a `sclx::cuda::task_scheduler` that guarantees a task is executed on a specific device in its own thread, useful when calling to external libraries like `cuBLAS` 

## API Changes

- `sclx::md_index_t` now has better method names for converting to/from linear indices
- Returned data pointer for `sclx::array::data()` is no longer constant by default.

## Bug Fixes
- Fixed memory leak issue in `sclx::detail::unified_ptr` that only occurred for exotic use cases
- Fixed issues with some APIs not working when using `sclx::array` data type was const-qualified
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
