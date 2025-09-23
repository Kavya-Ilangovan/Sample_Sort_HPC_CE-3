#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>

// Function to generate random integers
void generate_random_data(std::vector<int>& data, int num_elements) {
    for (int i = 0; i < num_elements; i++) {
        data[i] = rand() % 10000;
    }
}

// Function to merge two sorted vectors into one
std::vector<int> merge(const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> result(left.size() + right.size());
    int i = 0, j = 0, k = 0;

    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            result[k++] = left[i++];
        } else {
            result[k++] = right[j++];
        }
    }

    while (i < left.size()) {
        result[k++] = left[i++];
    }
    
    while (j < right.size()) {
        result[k++] = right[j++];
    }
    
    return result;
}

// Sample Sort Function
void sample_sort(int rank, int size, std::vector<int>& local_data, int num_elements) {
    int local_size = num_elements / size;

    // Local sorting
    std::sort(local_data.begin(), local_data.end());

    // Send and receive data using MPI_Send and MPI_Recv for merging
    std::vector<int> sorted_data;
    if (rank == 0) {
        sorted_data = local_data;
        for (int i = 1; i < size; i++) {
            std::vector<int> received_data(local_size);
            MPI_Recv(received_data.data(), local_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sorted_data = merge(sorted_data, received_data);
        }
        /*std::cout << "Sorted data by root: ";
        for (int i = 0; i < sorted_data.size(); i++) {
            std::cout << sorted_data[i] << " ";
        }
        std::cout << std::endl;*/
        std::cout<<"Sorted\n";
    } else {
        MPI_Send(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long num_elements = 10000000000;
    std::vector<int> data(num_elements);

    if (rank == 0) {
        generate_random_data(data, num_elements);
        /*std::cout << "Initial unsorted data: ";
        for (int i = 0; i < num_elements; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;*/
    }

    // Scatter the data to all processes
    int local_size = num_elements / size;
    std::vector<int> local_data(local_size);
    MPI_Scatter(data.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort the local data and merge using sample sort
    sample_sort(rank, size, local_data, num_elements);

    MPI_Finalize();
    return 0;
}

