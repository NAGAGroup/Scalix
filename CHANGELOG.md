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
