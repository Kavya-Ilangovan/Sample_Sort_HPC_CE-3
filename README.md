# MPI Sample Sort (Point-to-Point Implementation)

This repository contains an implementation of the **Sample Sort algorithm** using **MPI (Message Passing Interface)**.  
The implementation uses **only point-to-point communication primitives** (`MPI_Send`, `MPI_Recv`, `MPI_Isend`, `MPI_Irecv`) and avoids all collective operations (`MPI_Bcast`, `MPI_Scatterv`, `MPI_Alltoallv`, etc.) as required by the assignment.

---

##  Features
- Parallel sorting of large integer arrays across multiple MPI processes.
- **Point-to-point data distribution** (no `MPI_Scatter`).
- **Sample selection and pivoting** for balanced partitioning.
- **All-to-all bucket exchange** using only `MPI_Isend`/`MPI_Irecv`.
- **Local k-way merge** of received buckets into a globally sorted sub-array.
- Final **gather to root** and verification against a sequential sort.
- Timing instrumentation for major phases:
  - Distribution
  - Local sort
  - Pivot selection
  - Size exchange
  - Data exchange
  - Merge
  - Total runtime

---

##  Requirements
- C++11 or later
- MPI implementation (e.g., [OpenMPI](https://www.open-mpi.org/) or MPICH)

---

##  Build Instructions

Compile with:

```bash
mpic++ -O3 -std=c++11 -o sample_sort sample_sort_mpi.cpp
