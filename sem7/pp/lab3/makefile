### Данные варианта
TYPE=TYPE_DOUBLE
N=6300000
Q=26
K=3
EXTRA=33391
THREADS_COUNT="\"[3,9,12]\""
###
DEFINES= -D$(TYPE) -DN=$(N) -DQ=$(Q) -DK=$(K) -DTHREADS_COUNT=$(THREADS_COUNT) -DEXTRA=$(EXTRA)
CFLAGS= -std=c99 -O0 $(DEFINES)
CC=gcc
MPICC=mpicc

all: lab3mulmpi lab3nonmpi lab3mulmpiq lab3nonmpiq lab3omp lab3ompq

lab3mulmpi:
	$(MPICC) -o $@ mulmpi.c $(CFLAGS)

lab3nonmpi:
	$(MPICC) -o $@ nonmpi.c $(CFLAGS)

lab3mulmpiq:
	$(MPICC) -o $@ mulmpi.c $(CFLAGS) -DSLOWER

lab3nonmpiq:
	$(MPICC) -o $@ nonmpi.c $(CFLAGS) -DSLOWER

lab3omp:
	$(CC) -o $@ omp.c $(CFLAGS) -fopenmp

lab3ompq:
	$(CC) -o $@ omp.c $(CFLAGS) -DSLOWER -fopenmp

clean:
	rm lab3mulmpi
	rm lab3nonmpi
	rm lab3mulmpiq
	rm lab3nonmpiq
	rm lab3omp
	rm lab3ompq