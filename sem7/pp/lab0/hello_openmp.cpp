// OpenMP header
#include <cstdlib>
#include <omp.h>
#include <chrono>
#include <iostream>

int main(int argc, char* argv[])
{
	int nthreads = std::atoi(argv[1]);
	omp_set_num_threads(nthreads);
	int tid;

	// Begin of parallel region
	#pragma omp parallel private(nthreads, tid)
	{
		auto start = std::chrono::steady_clock::now();
		// Getting thread number
		tid = omp_get_thread_num();
		std::cout << "Hello world from thread " << tid << std::endl;

		if (tid == 0) {

			// Only master thread does this
			nthreads = omp_get_num_threads();
			std::cout << "Number of threads: " << nthreads << std::endl;
		}
		auto end = std::chrono::steady_clock::now();
		auto diff = end - start;
		std::cout << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
	}
}
