#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// Generate random data for local process
void generate_local_data(std::vector<int>& data, long local_n) {
    for (long i = 0; i < local_n; i++) {
        data[i] = rand() % 1000000; // values up to 1M
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 0: Get total array size from command line
    long total_N = 1000000; // default total array size
    if (argc > 1) total_N = atol(argv[1]);

    // Step 1: Calculate local chunk size for each process
    long base_chunk = total_N / size;
    long remainder = total_N % size;
    long local_n = base_chunk + (rank < remainder ? 1 : 0); // distribute remainder

    // Seed random differently per process
    srand(time(NULL) + rank * 100);

    // Step 2: Each process generates its local array
    std::vector<int> local_data(local_n);
    generate_local_data(local_data, local_n);

    if(rank==0) std::cout << "All processes generated local arrays.\n";

    MPI_Barrier(MPI_COMM_WORLD); // sync before timing
    double start_time = MPI_Wtime();

    // Step 3: Local sort
    std::sort(local_data.begin(), local_data.end());
    if(rank==0) std::cout << "Local sort done.\n";

    // Step 4: Pick local samples
    int s = size - 1; // number of samples per process
    std::vector<int> local_samples(s);
    for (int i = 0; i < s; i++) {
        long pos = (i+1) * local_n / size;
        if(pos >= local_n) pos = local_n - 1;
        local_samples[i] = local_data[pos];
    }

    // Step 5: Gather samples at root to compute pivots
    std::vector<int> gathered_samples;
    if(rank == 0) gathered_samples.resize(s * size);
    MPI_Gather(local_samples.data(), s, MPI_INT,
               gathered_samples.data(), s, MPI_INT,
               0, MPI_COMM_WORLD);

    // Step 6: Root chooses pivots
    std::vector<int> pivots(size-1);
    if(rank == 0) {
        std::sort(gathered_samples.begin(), gathered_samples.end());
        for(int i = 0; i < size-1; i++) {
            pivots[i] = gathered_samples[(i+1)*size - 1];
        }
        std::cout << "Pivots selected and broadcast.\n";
    }

    // Broadcast pivots to all processes
    MPI_Bcast(pivots.data(), size-1, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 7: Partition local data based on pivots using same buffer
    std::vector<std::vector<int>> buckets(size);
    int idx = 0;
    for(auto val : local_data) {
        while(idx < size-1 && val > pivots[idx]) idx++;
        buckets[idx].push_back(val);
    }

    if(rank==0) std::cout << "Partitioning done.\n";

    // Step 8: Exchange buckets among processes
    std::vector<int> send_counts(size), recv_counts(size);
    for(int i=0; i<size; i++) send_counts[i] = buckets[i].size();

    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> sdispls(size), rdispls(size);
    int send_total=0, recv_total=0;
    for(int i=0; i<size; i++) {
        sdispls[i] = send_total;
        rdispls[i] = recv_total;
        send_total += send_counts[i];
        recv_total += recv_counts[i];
    }

    std::vector<int> send_buf(send_total);
    int pos=0;
    for(int i=0; i<size; i++) {
        std::copy(buckets[i].begin(), buckets[i].end(), send_buf.begin()+pos);
        pos += buckets[i].size();
        buckets[i].clear(); // free memory
    }

    std::vector<int> recv_buf(recv_total);
    MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                  recv_buf.data(), recv_counts.data(), rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    if(rank==0) std::cout << "Redistribution done.\n";

    // Step 9: Final local sort
    std::sort(recv_buf.begin(), recv_buf.end());

    MPI_Barrier(MPI_COMM_WORLD); // sync before stopping timer
    double end_time = MPI_Wtime();

    if(rank == 0) {
        std::cout << "Final local sort done.\n";
        std::cout << "Memory-efficient sample sort finished.\n";
        std::cout << "Total array size = " << total_N << std::endl;
        std::cout << "Elapsed time (seconds) = " << (end_time - start_time) << std::endl;
    }

    MPI_Finalize();
    return 0;
}
