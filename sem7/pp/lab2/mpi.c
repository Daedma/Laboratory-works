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
	DECLARE_VECTORS(a);
#define MALLOC(var) var = (TYPE *)malloc(N * sizeof(TYPE))
	APPLY_FUNCTION(a, MALLOC);
#undef MALLOC
	if (proc_rank == 0)
	{
#define FILL(var) fill_array(var, N)
		APPLY_FUNCTION(a, FILL);
#undef FILL
	}

	// Время
	double ts = 0;          // последовательный алгоритм
	double tB = 0;          // широковещательная рассылка
	double tpp = 0, tR = 0; // параллельные алгоритмы
	for (int i = 0; i < 20; ++i) // внешний цикл для 20-ти повторений
	{
		TYPE sum;
		double st_time, end_time;

		// Последовательный алгоритм
		if (proc_rank == 0)
		{
			sum = 0;
			st_time = MPI_Wtime();
			for (int i = 0; i < N; ++i)
			{
				REPEAT_Q_TIMES
					sum = sum + APPLY_BIN_OP(a, i, +);
			}
			end_time = MPI_Wtime();
			ts += end_time - st_time;
			if (i == 19)
			{
				printf("Sequantional sum : " TYPE_FORMAT "\n", sum);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// Широковещательная рассылка и замер времени ее выполнения
		st_time = MPI_Wtime();
#define BROADCAST(var) MPI_Bcast(var, N, MPI_TYPE, 0, MPI_COMM_WORLD)
		APPLY_FUNCTION(a, BROADCAST);
#undef BROADCAST
		end_time = MPI_Wtime();
		tB += end_time - st_time;
		MPI_Barrier(MPI_COMM_WORLD);

		int k = N / proc_num;
		int i1 = k * proc_rank;
		int i2 = k * (proc_rank + 1);
		if (proc_rank == proc_num - 1)
			i2 = N;

		  // Параллельный алгоритм с использованием операций "точка-точка"
		st_time = MPI_Wtime();
		TYPE proc_sum = 0;
		for (int l = i1; l < i2; ++l)
		{
			REPEAT_Q_TIMES
				proc_sum += APPLY_BIN_OP(a, l, +);
		}
		TYPE total_sum = 0;
		if (proc_rank == 0)
		{
			total_sum = proc_sum;
			for (int j = 1; j < proc_num; ++j)
			{
				MPI_Recv(&proc_sum, 1, MPI_TYPE, j, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE);
				total_sum += proc_sum;
			}
		}
		else
		{
			MPI_Send(&proc_sum, 1, MPI_TYPE, 0, 0, MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		end_time = MPI_Wtime();
		tpp += end_time - st_time;
		if (proc_rank == 0 && i == 19)
		{
			printf("Point-to-point sum : " TYPE_FORMAT "\n", sum);
		}

		// Параллельный алгоритм с использованием коллективной операции
		st_time = MPI_Wtime();
		proc_sum = 0;
		for (int l = i1; l < i2; ++l)
		{
			REPEAT_Q_TIMES
				proc_sum += APPLY_BIN_OP(a, l, +);
		}
		MPI_Reduce(&proc_sum, &total_sum, 1, MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
		end_time = MPI_Wtime();
		tR += end_time - st_time;
		MPI_Barrier(MPI_COMM_WORLD);
		if (proc_rank == 0 && i == 19)
		{
			printf("Reduce sum : " TYPE_FORMAT "\n", sum);
		}
	}
	ts /= 20;
	tB /= 20;
	tpp /= 20;
	tR /= 20;

	// Вычисление ускорения
	double aRB = ts / (tB + tpp);
	double appB = ts / (tB + tR);
	double aR = ts / tpp;
	double ap = ts / tR;

	// Вывод полученных значений времени и ускорения
	if (proc_rank == 0)
	{
		printf("Sequential time: %.17f\n", ts);
		printf("Broadcast time: %.17f\n", tB);
		printf("Parallel time (point-to-point): %.17f\n", tpp);
		printf("Parallel time (reduce): %.17f\n", tR);
		printf("Speedup (point-to-point with broadcast): %.17f\n", aRB);
		printf("Speedup (reduce with broadcast): %.17f\n", appB);
		printf("Speedup (point-to-point without broadcast): %.17f\n", aR);
		printf("Speedup (reduce without broadcast): %.17f\n", ap);
	}

	// Освобождение памяти
	APPLY_FUNCTION(a, free);

	// Завершение MPI
	MPI_Finalize();
	return 0;
}
