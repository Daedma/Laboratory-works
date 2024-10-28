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
		printf("Threads count: %s\n", STR((THREADS_COUNT)));
		printf("Q: %d\n", Q);
		printf("Extra: %d\n", EXTRA);
	}

	// Создание массивов
	DECLARE_VECTORS(a); // Исходные вектора
	if (proc_rank == 0)
	{
#define MALLOC(x) x = (TYPE *)malloc((N + EXTRA) * sizeof(TYPE))
		APPLY_FUNCTION(a, MALLOC);
#undef MALLOC
#define FILL(x) fill_array(x, N + EXTRA)
		APPLY_FUNCTION(a, FILL);
#undef FILL
	}

	int base_elements = (N + EXTRA) / proc_num;
	int extra_elements = (N + EXTRA) % proc_num;

	// Вспомогательные массивы
	int* sendcounts = (int*)malloc(proc_num * sizeof(int));
	int* displs = (int*)malloc(proc_num * sizeof(int));

	// Инициализация sendcounts
	for (int i = 0; i < proc_num; ++i)
	{
		sendcounts[i] = base_elements + (i < extra_elements);
	}

	int k = sendcounts[proc_rank];

	// Инициализация displs
	displs[0] = 0;
	for (int i = 1; i < proc_num; ++i)
	{
		displs[i] = displs[i - 1] + sendcounts[i - 1];
	}

	DECLARE_VECTORS(a_local); // Обрабатываемые части
#define MALLOC(x) x = (TYPE *)malloc(k * sizeof(TYPE))
	APPLY_FUNCTION(a_local, MALLOC);
#undef MALLOC

	// Результирующие векторы
	TYPE* a_result;
	if (proc_rank == 0)
	{
		a_result = (TYPE*)malloc((N + EXTRA) * sizeof(TYPE));
	}
	TYPE* a_local_result = (TYPE*)malloc(k * sizeof(TYPE));

	// Время
	double ts = 0;   // последовательный алгоритм
	double tSc = 0;  // широковещательная рассылка
	double tnon = 0; // параллельные алгоритмы
	for (int j = 0; j < 20; ++j) // внешний цикл для 20-ти повторений
	{
		double st_time, end_time;

		// Последовательный алгоритм
		if (proc_rank == 0)
		{
			st_time = MPI_Wtime();
			for (int i = 0; i < N + EXTRA; ++i)
			{
				REPEAT_Q_TIMES
					a_result[i] = APPLY_BIN_OP(a, i, +);
			}
			end_time = MPI_Wtime();
			ts += end_time - st_time;
			if (j == 19)
			{
				print_vector("Sequational sum", a_result, N + EXTRA);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// Широковещательная рассылка и замер времени ее выполнения
		st_time = MPI_Wtime();
#define SCATTERV(x, y)  MPI_Scatterv(x, sendcounts, displs, MPI_TYPE, y, k, MPI_TYPE, 0, MPI_COMM_WORLD)
		APPLY_FUNCTION2(a, a_local, SCATTERV);
#undef SCATTERV
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
		MPI_Gatherv(a_local_result, k, MPI_TYPE, a_result, sendcounts, displs,
			MPI_TYPE, 0, MPI_COMM_WORLD);
		end_time = MPI_Wtime();
		tnon += end_time - st_time;
		MPI_Barrier(MPI_COMM_WORLD);
		if (proc_rank == 0 && j == 19)
		{
			print_vector("Parallel sum", a_result, N + EXTRA);
		}
	}
	ts /= 20;
	tSc /= 20;
	tnon /= 20;

	// Вычисление ускорения
	double amulSc = ts / (tSc + tnon);
	double amul = ts / tnon;

	// Вывод полученных значений времени и ускорения
	if (proc_rank == 0)
	{
		printf("Sequential time: %.17f\n", ts);
		printf("Scatter time: %.17f\n", tSc);
		printf("Parallel time (reduce): %.17f\n", tnon);
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
	free(sendcounts);
	free(displs);

	// Завершение MPI
	MPI_Finalize();
	return 0;
}