#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

int main(int argc, char* argv[])
{
// Вывод параметров индивидуального варианта
	printf("Type: %s\n", STR(TYPE));
	printf("K: %d\n", K);
	printf("N: %d\n", N);
	printf("Threads count: %s\n", THREADS_COUNT);
	printf("Q: %d\n", Q);

	// Создание массивов
	DECLARE_VECTORS(a);
#define MALLOC(var) var = (TYPE *)malloc(N * sizeof(TYPE))
	APPLY_FUNCTION(a, MALLOC);
#undef MALLOC
#define FILL(var) fill_array(var, N)
	APPLY_FUNCTION(a, FILL);
#undef FILL

	// Время
	double ts = 0; // последовательный алгоритм
	double tp = 0; // инициализация параллельной области
	double tc = 0, ta = 0, tr = 0; // параллельные алгоритмы

	for (int j = 0; j < 20; ++j) // внешний цикл для 20-ти повторений
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
		if (j == 19)
		{
			printf("Sequantional sum : " TYPE_FORMAT "\n", sum);
		}

		// Инициализация параллельной области
		st_time = omp_get_wtime();
#pragma omp parallel
		{
		  // Пустая параллельная область для замера времени инициализации
		}
		end_time = omp_get_wtime();
		tp += end_time - st_time;

		// critical
		sum = 0;
		st_time = omp_get_wtime();
#pragma omp parallel for
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
#pragma omp critical
			{
				sum = sum + APPLY_BIN_OP(a, i, +);
			}
		}
		end_time = omp_get_wtime();
		tc += end_time - st_time;
		if (j == 19)
		{
			printf("Critical sum : " TYPE_FORMAT "\n", sum);
		}

		// atomic
		sum = 0;
		st_time = omp_get_wtime();
#pragma omp parallel for
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
#pragma omp atomic
				sum = sum + APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		ta += end_time - st_time;
		if (j == 19)
		{
			printf("Atomic sum : " TYPE_FORMAT "\n", sum);
		}

		// reduction
		sum = 0;
		st_time = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
				sum = sum + APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		tr += end_time - st_time;
		if (j == 19)
		{
			printf("Reduction sum : " TYPE_FORMAT "\n", sum);
		}
	}
	// Расчет ускорения
	ts /= 20;
	tp /= 20;
	tc /= 20;
	ta /= 20;
	tr /= 20;

	// Вычисление ускорения
	double acp = ts / (tp + tc);
	double aap = ts / (tp + ta);
	double arp = ts / (tp + tr);
	double ac = ts / tc;
	double aa = ts / ta;
	double ar = ts / tr;

	// Вывод полученных значений времени и ускорения
	printf("Sequential time: %f\n", ts);
	printf("Parallel initialization time: %f\n", tp);
	printf("Parallel time (critical): %f\n", tc);
	printf("Parallel time (atomic): %f\n", ta);
	printf("Parallel time (reduction): %f\n", tr);
	printf("Speedup (critical with init): %f\n", acp);
	printf("Speedup (atomic with init): %f\n", aap);
	printf("Speedup (reduction with init): %f\n", arp);
	printf("Speedup (critical without init): %f\n", ac);
	printf("Speedup (atomic without init): %f\n", aa);
	printf("Speedup (reduction without init): %f\n", ar);

	// Освобождение памяти
	APPLY_FUNCTION(a, free);

	return 0;
}
