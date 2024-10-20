#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#ifndef TYPE
#define TYPE double
#endif

#ifdef TYPE
#if TYPE == double
#define MPI_TYPE MPI_DOUBLE
#elif TYPE == float
#define MPI_TYPE MPI_FLOAT
#elif TYPE == int
#define MPI_TYPE MPI_INT
#else
#error "Unsupported TYPE"
#endif
#else
#error "TYPE is not defined"
#endif

#ifndef K
#define K 3
#endif

#ifndef N
#define N 6300000
#endif

#ifndef Q
#define Q 26
#endif

#ifndef THREADS_COUNT
#define THREADS_COUNT [3, 9, 12]
#endif

#ifdef SLOWER
#define REPEAT_Q_TIMES for (int _rqt = 0; _rqt < Q; ++_rqt)
#else
#define REPEAT_Q_TIMES
#endif

#define CAT(a, b) CAT_IMPL(a, b)
#define CAT_IMPL(a, b) a ## b

#define VECTORS_1(var) TYPE* var##1;
#define VECTORS_2(var) VECTORS_1(var) TYPE* var##2;
#define VECTORS_3(var) VECTORS_2(var) TYPE* var##3;
#define VECTORS_4(var) VECTORS_3(var) TYPE* var##4;
#define VECTORS_5(var) VECTORS_4(var) TYPE* var##5;
#define VECTORS_6(var) VECTORS_5(var) TYPE* var##6;
#define VECTORS_7(var) VECTORS_6(var) TYPE* var##7;
#define VECTORS_8(var) VECTORS_7(var) TYPE* var##8;
#define VECTORS_9(var) VECTORS_8(var) TYPE* var##9;
#define VECTORS_10(var) VECTORS_9(var) TYPE* var##10;
#define DECLARE_VECTORS(var) CAT(VECTORS_, K)(var)

#define APPLY_BIN_OP_1(var, index, op) var##1[index]
#define APPLY_BIN_OP_2(var, index, op) (APPLY_BIN_OP_1(var, index, op) op var##2[index])
#define APPLY_BIN_OP_3(var, index, op) (APPLY_BIN_OP_2(var, index, op) op var##3[index])
#define APPLY_BIN_OP_4(var, index, op) (APPLY_BIN_OP_3(var, index, op) op var##4[index])
#define APPLY_BIN_OP_5(var, index, op) (APPLY_BIN_OP_4(var, index, op) op var##5[index])
#define APPLY_BIN_OP_6(var, index, op) (APPLY_BIN_OP_5(var, index, op) op var##6[index])
#define APPLY_BIN_OP_7(var, index, op) (APPLY_BIN_OP_6(var, index, op) op var##7[index])
#define APPLY_BIN_OP_8(var, index, op) (APPLY_BIN_OP_7(var, index, op) op var##8[index])
#define APPLY_BIN_OP_9(var, index, op) (APPLY_BIN_OP_8(var, index, op) op var##9[index])
#define APPLY_BIN_OP_10(var, index, op) (APPLY_BIN_OP_9(var, index, op) op var##10[index])
#define APPLY_BIN_OP(var, index, op) CAT(APPLY_BIN_OP_, K)(var, index, op)

#define APPLY_FUNCTION_1(var, func) (func(var##1));
#define APPLY_FUNCTION_2(var, func) APPLY_FUNCTION_1(var, func) (func(var##2));
#define APPLY_FUNCTION_3(var, func) APPLY_FUNCTION_2(var, func) (func(var##3));
#define APPLY_FUNCTION_4(var, func) APPLY_FUNCTION_3(var, func) (func(var##4));
#define APPLY_FUNCTION_5(var, func) APPLY_FUNCTION_4(var, func) (func(var##5));
#define APPLY_FUNCTION_6(var, func) APPLY_FUNCTION_5(var, func) (func(var##6));
#define APPLY_FUNCTION_7(var, func) APPLY_FUNCTION_6(var, func) (func(var##7));
#define APPLY_FUNCTION_8(var, func) APPLY_FUNCTION_7(var, func) (func(var##8));
#define APPLY_FUNCTION_9(var, func) APPLY_FUNCTION_8(var, func) (func(var##9));
#define APPLY_FUNCTION_10(var, func) APPLY_FUNCTION_9(var, func) (func(var##10));
#define APPLY_FUNCTION(var, func) CAT(APPLY_FUNCTION_, K)(var, func)

#define STR(x) STR_HELPER(x)
#define STR_HELPER(x) #x


void fill_array(TYPE* a, int size)
{
	for (int i = 0; i < size; ++i)
	{
		a[i] = (TYPE)1;
	}
}

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
	}

	// Создание массивов
	DECLARE_VECTORS(a);
#define MALLOC(var) var = (TYPE*)malloc(N * sizeof(TYPE))
	APPLY_FUNCTION(a, MALLOC);
	if (proc_rank == 0)
	{
#define FILL(var) fill_array(var, N)
		APPLY_FUNCTION(a, FILL);
	}

	// Время
	double ts = 0; // последовательный алгоритм
	double tB = 0; // широковещательная рассылка
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
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// Широковещательная рассылка и замер времени ее выполнения
		st_time = MPI_Wtime();
#define BROADCAST(var) MPI_Bcast(var, N, MPI_TYPE, 0, MPI_COMM_WORLD)
		APPLY_FUNCTION(a, BROADCAST);
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

		// Параллельный алгоритм с использованием коллективной операции
		st_time = MPI_Wtime();
		proc_sum = 0;
		for (int l = i1; l < i2; ++l)
		{
			REPEAT_Q_TIMES
				proc_sum += APPLY_BIN_OP(a, l, +);
		}
		MPI_Reduce(&proc_sum, &total_sum, 1, MPI_TYPE, MPI_SUM, 0,
			MPI_COMM_WORLD);
		end_time = MPI_Wtime();
		tR += end_time - st_time;
		MPI_Barrier(MPI_COMM_WORLD);
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
