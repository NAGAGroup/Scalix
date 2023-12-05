# Memory Regions Specification

## Introduction

In order to facilitate the distributed nature of the system in an efficient way,
memory regions are used to emulate paged memory. By paging the memory, we can
fine tune transfers between devices and process such that only data that is
going to be accessed will be transferred. By default, when reads are made from
`distributed_buffer` accessors, all pages of all memory regions in the compute
command will be updated. However, there will be APIs to allow for access
patterns (e.g. ranges and indices) to be specified which will in turn define the
pages in each region that will be updated.

## Data Types

- `mem_region` defines a region of memory associated with a device
- `mem_page` defines a page of memory associated with a memory region
- `page_array` a collection of pages that are associated with a memory region

The above will all be abstract types with a virtual interface that needs to be
implemented for each region type.

## Region Types

Since the base types are abstract, the below can be extended in the future to
include more region types or improve existing ones if developers building off of
this library find better ways to do things for their specific use cases. For
this reason, the `distributed_buffer` will likely take as an argument a region
factory of some sort that will be used to create the regions that will be used
by the `distributed_buffer`.

- `default_mem_region` is the most intuitive region in that the data is local to
  a process
- `distributed_reference_region` will be used as the "point-of-truth" for
  distributed buffers. Rather than making copies of data in the read-only
  regions of a `distributed_buffer`, this region will contain references to each
  region that contains "point-of-truth" data. The `page_array` returned by this
  region will contain pages with different owners. In order to interface with
  external APIs, "point-of-truth" data will need to be contiguous while still
  looking paged from the perspective of the `page_array` returned by the
  `default_mem_region`
- `mpi_mem_region` will be a dummy region that will interact with the
  `default_mem_region` in another process, pulling and sending data between the
  local process it is in and the remote process. The `page_array` returned by
  this region will not contain any data, instead only pulling and sending data
  when requested. This will be essential to avoid unnecessary overhead when
  copying to and from other regions where not all pages need to be updated. This
  region is low priority and will be implemented after the other regions are
  implemented. Implementing this will change API and ABI of the
  `default_mem_region`

## Default Region

This will be the most complex region in terms of data management since it is the
only one that truly owns any data.

### Data handles and Resource Locks

Data handles will be in charge of tracking the reference count to the paged
data. If the reference count for a page goes to zero, then the
`default_mem_region` object will need to delete the handle and the data. This
will be done internally, however, the user may want to keep data around that
they know will be used in the future, even if there is no reference to it. For
this a resource lock of some sort will need to be implemented.

## Copying Data Between Regions

Only allocated memory in the destination region will get copied. If there is an
allocated page in the destination region that is not allocated in the source,
this will raise an exception.
