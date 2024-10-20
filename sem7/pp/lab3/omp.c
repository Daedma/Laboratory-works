#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define CHUNK 1000

#ifndef TYPE
#define TYPE double
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
	if (argc < 2)
	{
		printf("Usage: %s <number_of_threads>", argv[0]);
		return -1;
	}

	int nthreads = atoi(argv[1]);
	omp_set_num_threads(nthreads);

	// Вывод параметров индивидуального варианта
	printf("Type: %s\n", STR(TYPE));
	printf("K: %d\n", K);
	printf("N: %d\n", N);
	printf("Threads count: %s\n", STR((THREADS_COUNT)));
	printf("Q: %d\n", Q);

	// Создание массивов
	DECLARE_VECTORS(a);
#define MALLOC(var) var = (TYPE*)malloc(N * sizeof(TYPE))
	APPLY_FUNCTION(a, MALLOC);
#define FILL(var) fill_array(var, N)
	APPLY_FUNCTION(a, FILL);

	// Время
	double ts = 0; // последовательный алгоритм
	double tp = 0; // инициализация параллельной области
	double tst = 0, td = 0, tg = 0; // параллельные алгоритмы

	for (int i = 0; i < 20; ++i) // внешний цикл для 20-ти повторений
	{
		TYPE sum;
		int i;
		double st_time, end_time;

		// Последовательный алгоритм
		sum = 0;
		st_time = omp_get_wtime();
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
				sum = sum + APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		ts += end_time - st_time;

		// Инициализация параллельной области
		st_time = omp_get_wtime();
#pragma omp parallel
		{
			// Пустая параллельная область для замера времени инициализации
		}
		end_time = omp_get_wtime();
		tp += end_time - st_time;

		sum = 0;
		st_time = omp_get_wtime();
#pragma omp parallel for reduction(+:sum) schedule(static, CHUNK)
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
				sum = sum + APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		tst += end_time - st_time;

		sum = 0;
		st_time = omp_get_wtime();
#pragma omp parallel for reduction(+:sum) schedule(dynamic, CHUNK)
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
				sum = sum + APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		td += end_time - st_time;

		sum = 0;
		st_time = omp_get_wtime();
#pragma omp parallel for reduction(+:sum) schedule(guided, CHUNK)
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
				sum = sum + APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		tg += end_time - st_time;
	}

	// Расчет ускорения
	ts /= 20;
	tp /= 20;
	tst /= 20;
	td /= 20;
	tg /= 20;

	// Вычисление ускорения
	double atsp = ts / (tp + tst);
	double adp = ts / (tp + td);
	double agp = ts / (tp + tg);
	double ast = ts / tst;
	double ad = ts / td;
	double ag = ts / tg;

	// Вывод полученных значений времени и ускорения
	printf("Sequential time: %f\n", ts);
	printf("Parallel initialization time: %f\n", tp);
	printf("Parallel time (static): %f\n", tst);
	printf("Parallel time (dynamic): %f\n", td);
	printf("Parallel time (guided): %f\n", tg);
	printf("Speedup (static with init): %f\n", atsp);
	printf("Speedup (dynamic with init): %f\n", adp);
	printf("Speedup (guided with init): %f\n", agp);
	printf("Speedup (static without init): %f\n", ast);
	printf("Speedup (dynamic without init): %f\n", ad);
	printf("Speedup (guided without init): %f\n", ag);

	// Освобождение памяти
	APPLY_FUNCTION(a, free);

	return 0;
}
