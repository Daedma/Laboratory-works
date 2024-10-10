#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define TYPE double
#define K 3
#define N 6300000
#define Q 26

void fill_array(TYPE* a, int size)
{
	for (int i = 0; i != size; ++i)
	{
		a[i] = (TYPE)rand() / RAND_MAX;
	}
}

TYPE sequential_sum(TYPE* a, int size)
{
	TYPE sum = 0;
	for (int i = 0; i != size; ++i)
	{
		sum += a[i];
	}
	return sum;
}

TYPE parallel_sum_reduction(TYPE* a, int size)
{
	TYPE sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i != size; ++i)
	{
		sum += a[i];
	}
	return sum;
}

TYPE parallel_sum_critical(TYPE* a, int size)
{
	TYPE sum = 0;
#pragma omp parallel
	{
		TYPE local_sum = 0;
#pragma omp for
		for (int i = 0; i != size; ++i)
		{
			local_sum += a[i];
		}
#pragma omp critical
		{
			sum += local_sum;
		}
	}
	return sum;
}

TYPE parallel_sum_atomic(TYPE* a, int size)
{
	TYPE sum = 0;
#pragma omp parallel for
	for (int i = 0; i != size; ++i)
	{
#pragma omp atomic
		sum += a[i];
	}
	return sum;
}

int main(int argc, char* argv[])
{
	int nthreads = atoi(argv[1]);
	omp_set_num_threads(nthreads);
// Вывод параметров индивидуального варианта
	printf("Type: %s\n", "double");
	printf("K: %d\n", K);
	printf("N: %d\n", N);
	printf("Threads count: [3, 9, 12]");
	printf("Q: %d\n", Q);

	// Создание массивов
	TYPE* a[K];
	for (int i = 0; i != K; ++i)
	{
		a[i] = (TYPE*)malloc(N * sizeof(TYPE));
		fill_array(a[i], N);
	}

	// Замер времени последовательного алгоритма
	double ts = 0;
	for (int i = 0; i != 20; ++i)
	{
		double st_time = omp_get_wtime();
		for (int j = 0; j != K; ++j)
		{
			sequential_sum(a[j], N);
		}
		double end_time = omp_get_wtime();
		ts += end_time - st_time;
	}
	ts /= 20;

	// Замер времени инициализации параллельной области
	double tp = 0;
	for (int i = 0; i != 20; ++i)
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
	double tc = 0, ta = 0, tr = 0;
	for (int i = 0; i != 20; ++i)
	{
		double st_time = omp_get_wtime();
		for (int j = 0; j < K; j++)
		{
			parallel_sum_critical(a[j], N);
		}
		double end_time = omp_get_wtime();
		tc += end_time - st_time;

		st_time = omp_get_wtime();
		for (int j = 0; j < K; j++)
		{
			parallel_sum_atomic(a[j], N);
		}
		end_time = omp_get_wtime();
		ta += end_time - st_time;

		st_time = omp_get_wtime();
		for (int j = 0; j < K; j++)
		{
			parallel_sum_reduction(a[j], N);
		}
		end_time = omp_get_wtime();
		tr += end_time - st_time;
	}
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
	for (int i = 0; i < K; i++)
	{
		free(a[i]);
	}

	return 0;
}
