#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define TYPE double
#define K 3
#define N 6300000
#define Q 26
#define CHUNK 1000

#ifdef SLOWER
#define REPEAT_Q_TIMES for (int _rqt = 0; _rqt < Q; ++_rqt)
#else
#define REPEAT_Q_TIMES
#endif


void fill_array(TYPE* a, int size)
{
	for (int i = 0; i < size; ++i)
	{
		a[i] = (TYPE)rand() / RAND_MAX;
	}
}

TYPE sequential_sum(TYPE* a, int size)
{
	TYPE sum = 0;
	for (int i = 0; i < size; ++i)
	{
		REPEAT_Q_TIMES sum += a[i];
	}
	return sum;
}

TYPE parallel_sum_static(TYPE* a, int size)
{
	TYPE sum = 0;
#pragma omp parallel for reduction(+:sum) shedule(static, CHUNK)
	for (int i = 0; i < size; ++i)
	{
		REPEAT_Q_TIMES sum += a[i];
	}
	return sum;
}

TYPE parallel_sum_dynamic(TYPE* a, int size)
{
	TYPE sum = 0;
#pragma omp parallel for reduction(+:sum) shedule(dynamic, CHUNK)
	for (int i = 0; i < size; ++i)
	{
		REPEAT_Q_TIMES sum += a[i];
	}
	return sum;
}

TYPE parallel_sum_guided(TYPE* a, int size)
{
	TYPE sum = 0;
#pragma omp parallel for reduction(+:sum) shedule(guided, CHUNK)
	for (int i = 0; i < size; ++i)
	{
		REPEAT_Q_TIMES sum += a[i];
	}
	return sum;
}

int main(int argc, char* argv[])
{
	int nthreads = atoi(argv[1]);
	omp_set_num_threads(nthreads);
// Вывод параметров индивидуального варианта
	printf("Type: %s\n", "double\n");
	printf("K: %d\n", K);
	printf("N: %d\n", N);
	printf("Threads count: [3, 9, 12]");
	printf("Q: %d\n", Q);

	// Создание массивов
	TYPE* a[K];
	for (int i = 0; i < K; ++i)
	{
		a[i] = (TYPE*)malloc(N * sizeof(TYPE));
		fill_array(a[i], N);
	}

	// Замер времени последовательного алгоритма
	double ts = 0;
	for (int i = 0; i < 20; ++i)
	{
		double st_time = omp_get_wtime();
		for (int j = 0; j < K; ++j)
		{
			sequential_sum(a[j], N);
		}
		double end_time = omp_get_wtime();
		ts += end_time - st_time;
	}
	ts /= 20;

	// Замер времени инициализации параллельной области
	double tp = 0;
	for (int i = 0; i < 20; ++i)
	{
		double st_time = omp_get_wtime();
#pragma omp parallel
		{
			// Пустая параллельная область для замера времени инициализации
		}
		double end_time = omp_get_wtime();
		tp += end_time - st_time;
	}
	tp /= 20;

	// Замер времени параллельных алгоритмов
	double tst = 0, td = 0, tg = 0;
	for (int i = 0; i < 20; ++i)
	{
		double st_time = omp_get_wtime();
		for (int j = 0; j < K; ++j)
		{
			parallel_sum_static(a[j], N);
		}
		double end_time = omp_get_wtime();
		tst += end_time - st_time;

		st_time = omp_get_wtime();
		for (int j = 0; j < K; ++j)
		{
			parallel_sum_dynamic(a[j], N);
		}
		end_time = omp_get_wtime();
		td += end_time - st_time;

		st_time = omp_get_wtime();
		for (int j = 0; j < K; ++j)
		{
			parallel_sum_guided(a[j], N);
		}
		end_time = omp_get_wtime();
		tg += end_time - st_time;
	}
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
	for (int i = 0; i < K; i++)
	{
		free(a[i]);
	}

	return 0;
}
