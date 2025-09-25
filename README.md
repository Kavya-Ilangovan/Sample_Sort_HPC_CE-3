# Parallel Sample Sort with MPI

## 📖 Overview
This repository implements the **Sample Sort algorithm** in C++ using the **Message Passing Interface (MPI)**.  
Two variants of the algorithm are included:  

1. **Point-to-Point Communication** (`samplempipp.cpp`) – uses only `MPI_Send` and `MPI_Recv` for sample gathering, pivot distribution, and bucket exchange.  
2. **Collective Communication** (`samplempi.cpp`) – uses collective operations such as `MPI_Gather`, `MPI_Bcast`, and `MPI_Alltoallv`.  

This project demonstrates the difference in complexity and efficiency between explicit point-to-point messaging and MPI collective communication.

---

## 📂 Files
- `samplempipp.cpp` → Point-to-point sample sort (exam requirement).  
- `samplempi.cpp` → Collective-based sample sort (for comparison).  

---

## ⚙️ Build Instructions
Compile both versions with `mpic++`:

```bash
mpic++ -O3 -std=c++11 -o samplempi samplempi.cpp
mpic++ -O3 -std=c++11 -o samplempipp samplempipp.cpp
