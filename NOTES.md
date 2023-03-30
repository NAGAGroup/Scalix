# Accessor Requirements for an Implementation (Data Consistency)

## Buffer Memory Layout

- typical metadata like size, type, etc.
- must maintain a point-of-truth for the data, but can have multiple copies of
  the data in various memory spaces. the rules for accessors below ensure that
  there are only read copies or a single write copy of the data at any given
  time. This maintains data consistency for the point-of-truth.
- the point-of-truth can and usually is spread across memory spaces.
- buffers should provide an api for setting the following hints:
  - read-mostly: used in UVM runtimes to telling the underlying driver how to
    manage the unified memory space
  - preferred memory space(s): used to force the point-of-truth to be in a
    specific memory space(s)
  - access via hint: can sometimes help the runtime to optimize the location of
    read-only copies of the data

## Memory chunks

- memory space tag (string, e.g. "node:0", "node:0:device:0", "node:0:uvm")
- typical metadata like size, type, etc.

## Host accessors

- Only one host exists for a given system, even one with multiple CPUs.
- If `const`, assumed read-only. Accesses will not signal to the owning buffer
  that it must update the point-of-truth data. Reads must be blocked if any
  device task is writing to the buffer until the task completes.
- If `non-const`, assumed read-write. Accesses will signal to the owning buffer
  that it must update the point-of-truth data as all accesses are assumed to be
  a write operation. This makes it important for users to use `const` accessors
  when possible. Accesses will be blocked while any device task using the buffer
  is running, even if the device is only reading from the buffer. The accessor
  must be updated with point-of-truth data after any device task that writes to
  the buffer completes.

## Device accessors

- Must implement `open` which notifies the owning buffer that it is to be used
  by a device task. The `open` call must wait until the buffer signals that it
  is safe to use the accessor. To allow overlapping `open` requests for
  different buffers used in a task, the `open` call must return a future.
- Must implement `close` which notifies the owning buffer that the device task
  is complete. It must also return a future. The buffer will use the `close`
  call to determine if any other accessors to the data are now safe to be used.
  It will also update the point-of-truth and any host accessors if the accessor
  was `non-const`

# Multiple CPU Systems Implementation?

- only one host
- everything else is either a node or a device
- devices are attached to nodes
- buffer memory management should be deterministic so that layout metadata is
  the same on all nodes

# General Sequence of Events

- create buffer, and runtime puts point-of-truth in a memory space(s)
- construct a gpu_task:
  - register inputs, each register call takes an optional read stencil to reduce
    the amount of data copied, empty access stencil indicates write-only
    operation
  - final setup: queries point-of-truth locations for the buffers associated
    with each input to determine the best memory space(s) to use for the task.
    this can be overridden by the user. sets up stencil mapping for each input
    if necessary (e.g. hash map from global index to local index)
- run the task when ready:
  - prepare: request access from buffers for each input
  - execute: run the task
  - finalize: each input is closed which notifies the buffer that it is no
    longer in use
