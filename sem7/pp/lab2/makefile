### Данные варианта
TYPE=TYPE_DOUBLE
N=6300000
Q=26
K=3
THREADS_COUNT="\"[3,9,12]\""
###
DEFINES= -D$(TYPE) -DN=$(N) -DQ=$(Q) -DK=$(K) -DTHREADS_COUNT=$(THREADS_COUNT)
CFLAGS= -std=c99 -O0 $(DEFINES)
CC=gcc
MPICC=mpicc

all: lab2mpi lab2mpiq lab2omp lab2ompq

lab2mpi:
	$(MPICC) -o $@ mpi.c $(CFLAGS)

lab2mpiq:
	$(MPICC) -o $@ mpi.c $(CFLAGS) -DSLOWER

lab2omp:
	$(CC) -o $@ omp.c $(CFLAGS) -fopenmp

lab2ompq:
	$(CC) -o $@ omp.c $(CFLAGS) -DSLOWER -fopenmp

clean:
	rm lab2mpi
	rm lab2mpiq
	rm lab2omp
	rm lab2ompq