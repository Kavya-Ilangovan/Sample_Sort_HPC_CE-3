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

    // Total array size
    long total_N = 1000000; // default
    if(argc > 1) total_N = atol(argv[1]);

    // Local chunk calculation
    long base_chunk = total_N / size;
    long remainder = total_N % size;
    long local_n = base_chunk + (rank < remainder ? 1 : 0);

    srand(time(NULL) + rank * 100);

    // Generate local data
    std::vector<int> local_data(local_n);
    generate_local_data(local_data, local_n);
    std::cout << "Rank " << rank << ": Local array generated (" << local_n << " elements)\n";

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Local sort
    std::sort(local_data.begin(), local_data.end());
    std::cout << "Rank " << rank << ": Local sort done\n";

    // Pick local samples
    int s = size - 1;
    std::vector<int> local_samples(s);
    for(int i=0; i<s; i++){
        long pos = (i+1) * local_n / size;
        if(pos >= local_n) pos = local_n-1;
        local_samples[i] = local_data[pos];
    }

    // -------------------- Point-to-Point Communication --------------------
    // Step 1: Send local samples to root (rank 0)
    std::vector<int> pivots(size-1);
    if(rank != 0){
        MPI_Send(local_samples.data(), s, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        // Root collects samples
        std::vector<int> gathered_samples;
        gathered_samples.insert(gathered_samples.end(), local_samples.begin(), local_samples.end());

        for(int src=1; src<size; src++){
            std::vector<int> temp(s);
            MPI_Recv(temp.data(), s, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            gathered_samples.insert(gathered_samples.end(), temp.begin(), temp.end());
        }

        // Sort and pick pivots
        std::sort(gathered_samples.begin(), gathered_samples.end());
        for(int i=0; i<size-1; i++){
            pivots[i] = gathered_samples[(i+1)*size -1];
        }
        std::cout << "Rank 0: Pivots selected\n";
    }

    // Step 2: Broadcast pivots manually
    for(int i=1; i<size; i++){
        if(rank == 0) {
            MPI_Send(pivots.data(), size-1, MPI_INT, i, 1, MPI_COMM_WORLD);
        } else if(rank == i){
            MPI_Recv(pivots.data(), size-1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Step 3: Partition local data into buckets
    std::vector<std::vector<int>> buckets(size);
    int idx = 0;
    for(auto val : local_data){
        while(idx<size-1 && val>pivots[idx]) idx++;
        buckets[idx].push_back(val);
    }
    std::cout << "Rank " << rank << ": Partitioning done\n";

    // Step 4: Exchange buckets using Send/Recv
    std::vector<int> send_counts(size), recv_counts(size);
    for(int i=0;i<size;i++) send_counts[i] = buckets[i].size();

    // Send counts first
    for(int i=0;i<size;i++){
        if(i!=rank){
            MPI_Send(&send_counts[i],1,MPI_INT,i,2,MPI_COMM_WORLD);
        }
    }

    // Receive counts from other processes
    for(int i=0;i<size;i++){
        if(i!=rank){
            MPI_Recv(&recv_counts[i],1,MPI_INT,i,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        } else {
            recv_counts[i] = send_counts[i];
        }
    }

    // Now exchange the actual data
    std::vector<int> recv_buf;
    int total_recv=0;
    for(int i=0;i<size;i++) total_recv += recv_counts[i];
    recv_buf.resize(total_recv);

    int offset=0;
    for(int i=0;i<size;i++){
        if(i!=rank){
            MPI_Send(buckets[i].data(), send_counts[i], MPI_INT, i, 3, MPI_COMM_WORLD);
        }
    }

    for(int i=0;i<size;i++){
        if(i!=rank){
            MPI_Recv(recv_buf.data()+offset, recv_counts[i], MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            std::copy(buckets[i].begin(), buckets[i].end(), recv_buf.begin()+offset);
        }
        offset += recv_counts[i];
    }
    std::cout << "Rank " << rank << ": Redistribution done\n";

    // Step 5: Final local sort
    std::sort(recv_buf.begin(), recv_buf.end());
    std::cout << "Rank " << rank << ": Final local sort done\n";

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if(rank==0){
        std::cout << "Point-to-point sample sort finished.\n";
        std::cout << "Total array size = " << total_N << std::endl;
        std::cout << "Elapsed time (seconds) = " << (end_time - start_time) << std::endl;
    }

    MPI_Finalize();
    return 0;
}