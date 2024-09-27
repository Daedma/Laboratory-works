#include <mpi.h>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
	auto start = std::chrono::steady_clock::now();

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::cout << "Hello world from rank " << world_rank << " out of " << world_size << "processors" << std::endl;
    auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	std::cout << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

    // Finalize the MPI environment.
    MPI_Finalize();
}