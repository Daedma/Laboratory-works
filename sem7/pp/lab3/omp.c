#include <omp.h>
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
#define FILL(var) fill_array(var, N)
	APPLY_FUNCTION(a, FILL);

	TYPE* a_result = (TYPE*)malloc(N * sizeof(TYPE));

	// Время
	double ts = 0; // последовательный алгоритм
	double tp = 0; // инициализация параллельной области
	double tst = 0, td = 0, tg = 0; // параллельные алгоритмы

	for (int k = 0; k < 20; ++k) // внешний цикл для 20-ти повторений
	{
		int i;
		double st_time, end_time;

		// Последовательный алгоритм
		st_time = omp_get_wtime();
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
				a_result[i] = APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		ts += end_time - st_time;
		if (k == 19)
		{
			print_vector("Sequational sum", a_result, N);
		}

		// Инициализация параллельной области
		st_time = omp_get_wtime();
#pragma omp parallel
		{
		  // Пустая параллельная область для замера времени инициализации
		}
		end_time = omp_get_wtime();
		tp += end_time - st_time;

		st_time = omp_get_wtime();
#pragma omp parallel for schedule(static, CHUNK)
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
				a_result[i] = APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		tst += end_time - st_time;
		if (k == 19)
		{
			print_vector("Static sum", a_result, N);
		}

		st_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, CHUNK)
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
				a_result[i] = APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		td += end_time - st_time;
		if (k == 19)
		{
			print_vector("Dynamic sum", a_result, N);
		}

		st_time = omp_get_wtime();
#pragma omp parallel for schedule(guided, CHUNK)
		for (i = 0; i < N; ++i)
		{
			REPEAT_Q_TIMES
				a_result[i] = APPLY_BIN_OP(a, i, +);
		}
		end_time = omp_get_wtime();
		tg += end_time - st_time;
		if (k == 19)
		{
			print_vector("Guided sum", a_result, N);
		}
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
	free(a_result);
	return 0;
}
