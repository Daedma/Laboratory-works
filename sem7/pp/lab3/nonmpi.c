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

#ifndef EXTRA
#define EXTRA 0
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

#define VECTORS_1(x) TYPE* x##1;
#define VECTORS_2(x) VECTORS_1(x) TYPE* x##2;
#define VECTORS_3(x) VECTORS_2(x) TYPE* x##3;
#define VECTORS_4(x) VECTORS_3(x) TYPE* x##4;
#define VECTORS_5(x) VECTORS_4(x) TYPE* x##5;
#define VECTORS_6(x) VECTORS_5(x) TYPE* x##6;
#define VECTORS_7(x) VECTORS_6(x) TYPE* x##7;
#define VECTORS_8(x) VECTORS_7(x) TYPE* x##8;
#define VECTORS_9(x) VECTORS_8(x) TYPE* x##9;
#define VECTORS_10(x) VECTORS_9(x) TYPE* x##10;
#define DECLARE_VECTORS(x) CAT(VECTORS_, K)(x)

#define APPLY_BIN_OP_1(x, index, op) x##1[index]
#define APPLY_BIN_OP_2(x, index, op) (APPLY_BIN_OP_1(x, index, op) op x##2[index])
#define APPLY_BIN_OP_3(x, index, op) (APPLY_BIN_OP_2(x, index, op) op x##3[index])
#define APPLY_BIN_OP_4(x, index, op) (APPLY_BIN_OP_3(x, index, op) op x##4[index])
#define APPLY_BIN_OP_5(x, index, op) (APPLY_BIN_OP_4(x, index, op) op x##5[index])
#define APPLY_BIN_OP_6(x, index, op) (APPLY_BIN_OP_5(x, index, op) op x##6[index])
#define APPLY_BIN_OP_7(x, index, op) (APPLY_BIN_OP_6(x, index, op) op x##7[index])
#define APPLY_BIN_OP_8(x, index, op) (APPLY_BIN_OP_7(x, index, op) op x##8[index])
#define APPLY_BIN_OP_9(x, index, op) (APPLY_BIN_OP_8(x, index, op) op x##9[index])
#define APPLY_BIN_OP_10(x, index, op) (APPLY_BIN_OP_9(x, index, op) op x##10[index])
#define APPLY_BIN_OP(x, index, op) CAT(APPLY_BIN_OP_, K)(x, index, op)

#define APPLY_FUNCTION_1(x, func) (func(x##1));
#define APPLY_FUNCTION_2(x, func) APPLY_FUNCTION_1(x, func) (func(x##2));
#define APPLY_FUNCTION_3(x, func) APPLY_FUNCTION_2(x, func) (func(x##3));
#define APPLY_FUNCTION_4(x, func) APPLY_FUNCTION_3(x, func) (func(x##4));
#define APPLY_FUNCTION_5(x, func) APPLY_FUNCTION_4(x, func) (func(x##5));
#define APPLY_FUNCTION_6(x, func) APPLY_FUNCTION_5(x, func) (func(x##6));
#define APPLY_FUNCTION_7(x, func) APPLY_FUNCTION_6(x, func) (func(x##7));
#define APPLY_FUNCTION_8(x, func) APPLY_FUNCTION_7(x, func) (func(x##8));
#define APPLY_FUNCTION_9(x, func) APPLY_FUNCTION_8(x, func) (func(x##9));
#define APPLY_FUNCTION_10(x, func) APPLY_FUNCTION_9(x, func) (func(x##10));
#define APPLY_FUNCTION(x, func) CAT(APPLY_FUNCTION_, K)(x, func)

#define APPLY_FUNCTION2_1(x, y, func) (func(x##1, y##1));
#define APPLY_FUNCTION2_2(x, y, func) APPLY_FUNCTION2_1(x, y, func) (func(x##2, y##2));
#define APPLY_FUNCTION2_3(x, y, func) APPLY_FUNCTION2_2(x, y, func) (func(x##3, y##3));
#define APPLY_FUNCTION2_4(x, y, func) APPLY_FUNCTION2_3(x, y, func) (func(x##4, y##4));
#define APPLY_FUNCTION2_5(x, y, func) APPLY_FUNCTION2_4(x, y, func) (func(x##5, y##5));
#define APPLY_FUNCTION2_6(x, y, func) APPLY_FUNCTION2_5(x, y, func) (func(x##6, y##6));
#define APPLY_FUNCTION2_7(x, y, func) APPLY_FUNCTION2_6(x, y, func) (func(x##7, y##7));
#define APPLY_FUNCTION2_8(x, y, func) APPLY_FUNCTION2_7(x, y, func) (func(x##8, y##8));
#define APPLY_FUNCTION2_9(x, y, func) APPLY_FUNCTION2_8(x, y, func) (func(x##9, y##9));
#define APPLY_FUNCTION2_10(x, y, func) APPLY_FUNCTION2_9(x, y, func) (func(x##10, y##10));
#define APPLY_FUNCTION2(x, y, func) CAT(APPLY_FUNCTION2_, K)(x, y, func)

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
		printf("Extra: %d\n", EXTRA);
	}

	// Создание массивов
	DECLARE_VECTORS(a); // Исходные вектора
	if (proc_rank == 0)
	{
#define MALLOC(x) x = (TYPE*)malloc((N + EXTRA) * sizeof(TYPE))
		APPLY_FUNCTION(a, MALLOC);
#define FILL(x) fill_array(x, N + EXTRA)
		APPLY_FUNCTION(a, FILL);
	}

	int extra_elements = (N + EXTRA) % proc_num; // Дополнительные элементы
	int k = (N + EXTRA) / proc_num + (proc_rank < extra_elements); // Количество обрабатываемых этим процессом элементов
	DECLARE_VECTORS(a_local); // Обрабатываемые части
#define MALLOC(x) x = (TYPE*)malloc(k * sizeof(TYPE))
	APPLY_FUNCTION(a_local, MALLOC);

	// Результирующие векторы
	TYPE* a_result;
	if (proc_rank == 0)
	{
		a_result = (TYPE*)malloc((N + EXTRA) * sizeof(TYPE));
	}
	TYPE* a_local_result = (TYPE*)malloc(k * sizeof(TYPE));

	// Вспомогательные массивы
	int* sendcounts = (int*)malloc(proc_num * sizeof(int));
	int* displs = (int*)malloc(proc_num * sizeof(int));

	// Инициализация sendcounts
	for (int i = 0; i < proc_num; ++i)
	{
		sendcounts[i] = k + (i < extra_elements);
	}

	// Инициализация displs
	displs[0] = 0;
	for (int i = 1; i < proc_num; ++i)
	{
		displs[i] = displs[i - 1] + sendcounts[i - 1];
	}


	// Время
	double ts = 0; // последовательный алгоритм
	double tSc = 0; // широковещательная рассылка
	double tnon = 0; // параллельные алгоритмы
	for (int i = 0; i < 20; ++i) // внешний цикл для 20-ти повторений
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
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// Широковещательная рассылка и замер времени ее выполнения
		st_time = MPI_Wtime();
#define SCATTER(x, y) MPI_Scatterv(x, sendcounts, displs, MPI_TYPE, y, k, MPI_TYPE, 0, MPI_COMM_WORLD)
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
		MPI_Gatherv(a_local_result, k, MPI_TYPE, a_result, sendcounts, displs, MPI_TYPE, 0, MPI_COMM_WORLD);
		end_time = MPI_Wtime();
		tnon += end_time - st_time;
		MPI_Barrier(MPI_COMM_WORLD);
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