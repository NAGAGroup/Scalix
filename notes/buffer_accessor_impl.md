# Buffer-Accessor Implementation

<!--toc:start-->

- [Buffer-Accessor Implementation](#buffer-accessor-implementation)
  - [Buffer Responsibilities](#buffer-responsibilities)
    - [Allocating Partitions and Device-Side Pointers via Strategies](#allocating-partitions-and-device-side-pointers-via-strategies)
  - [Queue Responsibilities](#queue-responsibilities)
  - [Command Handler Responsibilities](#command-handler-responsibilities)
  <!--toc:end-->

## Buffer Responsibilities

- Allocate device-side pointers to buffer partitions, return accessor (allocation details in [Allocating Partitions and Device-Side Pointers via Strategies](#allocating-partitions-and-device-side-pointers-via-strategies))
- Generate task to make partition data in the accessor valid (fetch data from valid buffer partitions)
- Using access mode, assign dependencies to the data fetch task
- Request fetch wait task from `cgh` and assign data fetch task as a dependency
- Create task that updates all active partitions in buffer for any writes made during access (push data back to active buffer partitions)
- Request command wait task from `cgh` and assign as dependency to push data task
- Update task metadata in buffer with push data task so that it can be used as a dependency in subsequent commands
- Request push wait task from `cgh` and assign push data task as a dependency
- Register partitions with `cgh` which will ensure partitions are not deleted, even if buffer goes out of all scopes

### Allocating Partitions and Device-Side Pointers via Strategies

A strategy defines the specific access required for a command, or more specifically, the partitioned commands. The default strategy is to make all partitions available to all sub-commands.

This is not recommended, however, as it can lead to sub-commands that try to write to the same buffer entry at the same time. Instead, the user should define a strategy that only makes the partitions available to the sub-commands that need them.

Because of the strategy pattern and also because the more fine-grained strategies will require information on the user provided parameters passed to the command APIs, which is called after the `get_access` APIs, we need to add a few more
responsibilities to the buffer:

- Request future from `cgh` that returns type of command API that was called and the arguments/command config for the command API so that the strategy can be applied
- Request the allocation task from the strategy and assign it as a dependency to the data fetch task

## Queue Responsibilities

- Create fetch wait task
- Create command wait task
- Create push wait task
- Create vector of void smart pointers which will be used to store data partitions and keep their reference count from going to zero, avoiding the early cleanup of data before a command has completed
- Assign all the above to a global metadata object
  - The metadata object should include a method that allows each `cgh` to notify it that the problem setup for the calling `cgh` is completed. Once all `cgh` objects call this notify, the metadata object can then safely launch all the tasks
  - The device split info of the queue should also be assigned to this metadata so that the `cgh` objects know how to split up calls like `parallel_for`
- For each device in the queue:
  - Call `sycl::queue::submit`
  - Create `cgh` and assign native `sycl::command_handler`, global metadata and native `sycl::queue` to the `cgh`
  - Using the future returned by the fetch wait task, create a `sycl::event` that is assigned as a dependent event to the native `sycl::command_handler`
  - pass the `cgh` to the user-defined functor defining their command submission
- From the future returned by the push wait task, create an event and return it from the `sclx::queue::submit` APIs

## Command Handler Responsibilities

- Provide the required APIs for the buffer to implement its responsibilities
- Implement command APIs like `parallel_for`
- Once a command API is called, notify the global metadata object that problem setup for this `cgh` is completed
