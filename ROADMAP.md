# First Stable Release: 1.0.0 = CUDA Unified Memory

We will be following semantic versioning so 0.x.x releases will be considered unstable and may contain breaking changes.
1.0.0 will indicate our first stable release.

The aim for 1.x.x is the Unified Memory implementation, limited to a single CPU with multiple GPUS. At some point, a
feature freeze will occur and 1.x.x will only receive bug fixes from the official team (if the team grows, otherwise
it'll just be me). New features will still be accepted via PRs, however, or if there is enough community consensus to
add it. Version 1 is mostly a proof of concept, after which version 2 will begin development.

# Second Stable Release: 2.0.0 = SYCL(or CUDA?) + MPI

Versions 2.x.x will be focused on expanding our distributed device abstraction to a true distributed architecture,
likely implemented with SYCL+MPI. We may use CUDA, depending on the state of the CUDA backend for
the [oneAPI OneMKL library](https://github.com/oneapi-src/oneMKL). This version will have drastically different API,
likely utilizing the buffer-accessor model, as we can no longer rely on a unified memory model under the hood. The
overall concepts from Version 1, however,
will be maintained (e.g. load distribution, problem definition requirements, user abstractions, etc.). We already have
come up with the general idea of how this would work, but We want to work out all the kinks first with the
easier-to-manage unified memory model on a single CPU. Version 2 will not replace, Version 1, as Version 1 will still
likely perform much better on single CPU systems. This is why Version 1 will continue to receive bug fixes and community
supplied features for the foreseeable future.

# Third Stable Release: 3.0.0 = Generalized Specification!

This leads us to Version 3. Version 3 will take Version 2 and generalize it to a specification, rather than any sort of
specific implementation. Our reference implementation will consist of two backends, modifying both Version 1 and Version
2 to fit the specification. At this point Version 1 and Version 2 will both be sunset, no longer receiving any official
updates from the development team.

At this point our implementation and specification will separate their versioning. The specification will continue from
3.x.x, while the implementation will move to a separate repo and versioning cadence. This will allow the implementation
to explore new ideas and features, potentially introducing breaking changes, without affecting the specification.
Of course these breaking changes won't affect spec compliant APIs, but instead will be extensions to the specification.
Extensions from the implementation may or may not get absorbed into the specification, at which point the specification
version will be updated accordingly.

# What Happens to the Old Versions?

We expect Version 2 to be extremely close in API to Version 3, with the main differences coming from Version 3 needing
to support multiple drop-in replacements. For this reason, we will likely fully drop Version 2 and require users to
upgrade to Version 3.

Version 1 is special in that, being a single CPU implementation, has more relaxed requirements for implementing a
correct algorithm. We expect that some users may prefer this simplicity and will continue to use Version 1. For this
reason, depending on community support and the size of the development team, version one may fork into a separate
official project. Of course, nothing is stopping anyone from forking the project and continuing to develop Version 1
themselves.