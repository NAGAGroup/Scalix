# API Specification for Backend

## Change to Allocation Process

Currently I have a pretty odd setup for an `allocation_factory` type object,
that holds onto pages, and then pages have references to the allocations. This
is odd and a waste of space, so instead, `access_anchors` will store strong
references to the pages, so if the access_anchor, or any of its shared
references, are in scope, the data will be in scope.

Since we do not want page size to be visible in the user-facing types,
`access_anchor` will be opaque, but will still include type and rank
information, just to increase chances of catching bugs at compile time. The
anchor will also contain the pointer to the buffer for which it is an anchor, so
the buffer doesn't go out of scope while the anchor is still in scope. The
buffer should be accessible through the anchor, but not the other way around.

Access anchors can be used with multiple queues. The anchor will similarly
maintain the pointer to each queue it's been used with, and will obviously keep
the data around. The user will be able to iterate through the queues.

A check will be done at runtime to ensure the anchor is being called on the
right buffer, ensuring that diving into the APIs for the buffer page size is
safe.

Again, to catch as many bugs as possible at compile time, the anchor will wrap
an allocation that does NOT have page info abstracted away. The queue/command
backend will have it's own special access anchor that will be used to capture
and maintain references to all pages being used in the queue, whether those
allocations are one-offs for the specific `submit` call, or if they are being
provided by a user-defined access anchor.

## So What Does This Look Like?

Starting from the bottom and working up, we go over all the types I've thought
of so far.

### Page

There are two categories of pages I've thought of so far:

1. Local pages, these exist on the physical node. These can be thought of as
   default, and will be all I implement for the first version of the library.
2. Remote pages, these exist on a different node. These are proxies that
   interact with the local pages on the remote and return futures for data and
   other things.

The above two categories may have further subcategories. Device vs. host/node
memory is one example. Currently, that separation isn't necessary, but I've
aliased the `local_page` type to `device_page` and `host_page` so future code
changes will be easier.

With all these categories, it makes sense to go with an abstract base class for
pages, as they will not be directly accessed on any device, so we get access to
virtual functions!

The interface should allow for:

1. Quickly testing if page is local or remote
2. Access the data, local should be immediate, remote should be a future
3. Set/get the write bit, local should be immediate, remote should be a future.
   Not sure if this is needed for remote pages. But page tables from different
   devices will be used to update other page tables, which indicates a write.
   So, I think this is needed.
4. Get the page index. Immediate for both local and remote, I believe. Since
   page replacements will have events associated with them. And all MPI should
   be synced up by end of event.
5. Copy from another page, page can be from the same device or a different
   device. Copies should be asynchronous, returning a `sycl::event`

### Page Handle

I don't want to pass pointers around, even smart ones, nor do I want to make
sure everything can be passed by reference. So, to avoid slicing and to make
sure everything is passed by reference, I've created a `page_handle` type that
contains a pointer to the page. The handle exposes the interface of the page, as
well as some additional APIs for higher level operations.

The page handle comes in two flavors, weak and strong. We want to be able to
store weak references to pages in the actual buffer management system so data is
cleaned up unless the user has access anchors, or if an implicit anchor created
by a queue is still in scope. Strong handles will only be stored by a single
access anchor (or shared references to the anchor), and will be used by the
backend to actually access the data. The weak handle has a very limited API so
that we can be sure that page manipulation and data access is only done through
the strong handle, which has an API for checking validity.

**Weak Handle API**

-
