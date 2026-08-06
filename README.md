# Distributed Tridiagonal Solver

This repository implements the distributed batched tridiagonal solver described
in the `docs/knowledge` submodule.

Milestone A is an ordinary CPU numerical reference. Later milestones introduce
Kokkos and MPI. The current implementation contains only the canonical batch
data model; the global Thomas solver and Diez algorithms are not yet
implemented.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
