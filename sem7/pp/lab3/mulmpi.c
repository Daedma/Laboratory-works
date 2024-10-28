#include "mpi.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

int main(int argc, char* argv[])
{
// Инициализация MPI
	MPI_Init(&argc, &argv);
	int proc_rank, proc_num;
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	if (proc_rank == 0)
	{
// Вывод параметров индивидуального варианта
		printf("Type: %s\n", STR(TYPE));
		printf("K: %d\n", K);
		printf("N: %d\n", N);
		printf("Threads count: %s\n", THREADS_COUNT);
		printf("Q: %d\n", Q);
	}

	// Создание массивов
	DECLARE_VECTORS(a); // Исходные вектора
	if (proc_rank == 0)
	{
#define MALLOC(x) x = (TYPE *)malloc(N * sizeof(TYPE))
		APPLY_FUNCTION(a, MALLOC);
#define FILL(x) fill_array(x, N)
		APPLY_FUNCTION(a, FILL);
	}

	int k = N / proc_num; // Количество обрабатываемых каждым процессом элементов
	DECLARE_VECTORS(a_local); // Обрабатываемые части
#define MALLOC(x) x = (TYPE *)malloc(k * sizeof(TYPE))
	APPLY_FUNCTION(a_local, MALLOC);

	// Результирующие векторы
	TYPE* a_result;
	if (proc_rank == 0)
	{
		a_result = (TYPE*)malloc(N * sizeof(TYPE));
	}
	TYPE* a_local_result = (TYPE*)malloc(k * sizeof(TYPE));

	// Время
	double ts = 0;   // последовательный алгоритм
	double tSc = 0;  // широковещательная рассылка
	double tmul = 0; // параллельные алгоритмы
	for (int k = 0; k < 20; ++k) // внешний цикл для 20-ти повторений
	{
		double st_time, end_time;

		// Последовательный алгоритм
		if (proc_rank == 0)
		{
			st_time = MPI_Wtime();
			for (int i = 0; i < N; ++i)
			{
				REPEAT_Q_TIMES
					a_result[i] = APPLY_BIN_OP(a, i, +);
			}
			end_time = MPI_Wtime();
			ts += end_time - st_time;
			if (k == 19)
			{
				print_vector("Sequentional sum", a_result, N);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// Широковещательная рассылка и замер времени ее выполнения
		st_time = MPI_Wtime();
#define SCATTER(x, y) MPI_Scatter(x, k, MPI_TYPE, y, k, MPI_TYPE, 0, MPI_COMM_WORLD)
		APPLY_FUNCTION2(a, a_local, SCATTER);
		end_time = MPI_Wtime();
		tSc += end_time - st_time;
		MPI_Barrier(MPI_COMM_WORLD);

		// Параллельный алгоритм сложения векторов
		st_time = MPI_Wtime();
		for (int l = 0; l < k; ++l)
		{
			REPEAT_Q_TIMES
				a_local_result[l] = APPLY_BIN_OP(a_local, l, +);
		}
		MPI_Gather(a_local_result, k, MPI_TYPE, a_result, k, MPI_TYPE, 0,
			MPI_COMM_WORLD);
		end_time = MPI_Wtime();
		tmul += end_time - st_time;
		MPI_Barrier(MPI_COMM_WORLD);
		if (proc_rank == 0 && k == 19)
		{
			print_vector("Parallel sum", a_result, N);
		}
	}
	ts /= 20;
	tSc /= 20;
	tmul /= 20;

	// Вычисление ускорения
	double amulSc = ts / (tSc + tmul);
	double amul = ts / tmul;

	// Вывод полученных значений времени и ускорения
	if (proc_rank == 0)
	{
		printf("Sequential time: %.17f\n", ts);
		printf("Scatter time: %.17f\n", tSc);
		printf("Parallel time (reduce): %.17f\n", tmul);
		printf("Speedup (reduce with scatter): %.17f\n", amulSc);
		printf("Speedup (reduce without scatter): %.17f\n", amul);
	}

	// Освобождение памяти
	if (proc_rank == 0)
	{
		APPLY_FUNCTION(a, free);
		free(a_result);
	}
	APPLY_FUNCTION(a_local, free);
	free(a_local_result);

	// Завершение MPI
	MPI_Finalize();
	return 0;
}
