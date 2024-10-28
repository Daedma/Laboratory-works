#ifndef CONFIG_H
#define CONFIG_H

#if defined(TYPE_DOUBLE)
#define TYPE double
#define TYPE_FORMAT "%f"
#elif defined(TYPE_FLOAT)
#define TYPE float
#define TYPE_FORMAT "%f"
#elif defined(TYPE_INT)
#define TYPE int
#define TYPE_FORMAT "%d"
#else
#define TYPE double
#define TYPE_FORMAT "%f"
#endif

#ifndef K
#define K 3
#endif

#ifndef DIMDIV
#define DIMDIV 1
#endif

#ifndef N
#define N 6300000/DIMDIV
#endif

#ifndef DIMDIV1
#define DIMDIV1 3
#endif

#ifndef DIMDIV2
#define DIMDIV2 9
#endif

#ifndef GRID_SIZE
#define GRID_SIZE 1
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#endif