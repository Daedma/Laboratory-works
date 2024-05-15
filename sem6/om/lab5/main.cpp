// ЛР 5: Симплекс метод прямой и двухэтапный

#include <iostream>
#include <functional>
#include <cmath>
#include <utility>
#include <numeric>
// #include "Mathter/Vector.hpp"
// #include "Mathter/Matrix.hpp"
// #include "Mathter/IoStream.hpp"
#include <cstring>
#include <cassert>

typedef double num_t;

struct matrix_t
{
	num_t* pdata;
	size_t nrow;
	size_t ncol;
};

struct vector_t
{
	num_t* pdata;
	size_t size;
};

vector_t vnew(size_t size)
{
	vector_t v;
	v.pdata = new num_t[size];
	v.size = size;
	return v;
}

num_t& vget(vector_t* v, size_t ind)
{
	assert(ind < v->size);
	return v->pdata[ind];
}

num_t* vpget(vector_t* v, size_t ind)
{
	assert(ind < v->size);
	return v->pdata + ind;
}

const num_t& vget(const vector_t* v, size_t ind)
{
	assert(ind < v->size);
	return v->pdata[ind];
}

const num_t* vpget(const vector_t* v, size_t ind)
{
	assert(ind < v->size);
	return v->pdata + ind;
}

matrix_t mnew(size_t nrow, size_t ncol)
{
	matrix_t m;
	m.pdata = new num_t[nrow * ncol];
	m.nrow = nrow;
	m.ncol = ncol;
	return m;
}

num_t& mget(matrix_t* m, size_t i, size_t j)
{
	assert(i < m->nrow && j < m->ncol);
	return m->pdata[i * m->ncol + j];
}

num_t* mpget(matrix_t* m, size_t i, size_t j)
{
	return m->pdata + i * m->ncol + j;
}

const num_t& mget(const matrix_t* m, size_t i, size_t j)
{
	assert(i < m->nrow && j < m->ncol);
	return m->pdata[i * m->ncol + j];
}

const num_t* mpget(const matrix_t* m, size_t i, size_t j)
{
	return m->pdata + i * m->ncol + j;
}

num_t* mmassign(matrix_t* dest, const matrix_t* src, size_t i, size_t j)
{
	assert(src->nrow + i < dest->nrow && src->ncol + j < dest->ncol);
	for (size_t is = 0; is != src->nrow; ++i, ++is)
	{
		memcpy(dest->pdata + i * dest->ncol + j, src->pdata + is * src->ncol, sizeof(num_t) * src->ncol);
	}
	return mpget(dest, src->nrow + i + 1, src->ncol + j);
}

num_t* mrassign(matrix_t* dest, const vector_t* src, size_t i, size_t j)
{
	assert(src->size + j < dest->ncol && i < dest->nrow);
	memcpy(dest->pdata + i * dest->ncol + j, src->pdata, sizeof(num_t) * src->size);
	return dest->pdata + i * dest->ncol + j + 1;
}

num_t* mcassign(matrix_t* dest, const vector_t* src, size_t i, size_t j)
{
	assert(src->size + i < dest->nrow && j < dest->ncol);
	for (size_t is = 0; is != src->size; ++is)
	{
		mget(dest, i + is, j) = vget(src, is);
	}
	return dest->pdata + (i + 1) * dest->ncol + j;
}


struct lpproblem
{
	matrix_t areaa;
	vector_t areab;
	vector_t funcc;
};

struct lpsolution
{
	vector_t point;
	num_t val;
	int rcode;
};

void init_table(const lpproblem* pdata, matrix_t* pout)
{
	assert(pdata->areaa.ncol == pdata->funcc.size && pdata->areaa.nrow == pdata->areab.size);
	*pout = mnew(pdata->areaa.nrow + 1, pdata->areaa.ncol + pdata->areab.size + 1);
	memset(pout->pdata, 0x0, sizeof(num_t) * pout->ncol * pout->nrow);
	mmassign(pout, &pdata->areaa, 0, 0);
	mrassign(pout, &pdata->funcc, pdata->areaa.nrow, 0);
	for (num_t* i = mpget(pout, pdata->areaa.nrow, 0),
		*end = mpget(pout, pdata->areaa.nrow, pdata->areaa.ncol + 1); i != end; ++i)
	{
		*i = -(*i);
	}
	mcassign(pout, &pdata->areab, 0, pdata->areaa.ncol + pdata->areab.size);
	for (size_t i = pdata->areaa.ncol; i != pdata->areaa.nrow; ++i)
	{
		mget(pout, i, i) = 1;
	}
}

void init_basis(const lpproblem* pdata, vector_t* pout)
{
	*pout = vnew(pdata->areaa.nrow);
	for (size_t i = 0; i != pout->size; ++i)
	{
		pout->pdata[i] = i + pdata->areaa.ncol;
	}
}

bool simplex_iteration(matrix_t* ptable, vector_t* pbasis)
{
 // TODO
}

void write_solution(const matrix_t* ptable, const vector_t* pbasis, lpsolution* ppout)
{
 // TODO
}

void lpsolve(const lpproblem* pdata, lpsolution* pout)
{
	matrix_t table;
	init_table(pdata, &table);
	vector_t basis;
	init_basis(pdata, &basis);
	while (simplex_iteration(&table, &basis));
	write_solution(&table, &basis, pout);
}

// using value_type = double;

// template<int N>
// using Vector = mathter::Vector<value_type, N>;

// template<int Rows, int Columns>
// using Matrix = mathter::Matrix<value_type, Rows, Columns, mathter::eMatrixOrder::PRECEDE_VECTOR>;

// template<int NVariables, int NConditions>
// Vector<NVariables> simplex_method(const Vector<NVariables>& c, const Matrix<NConditions, NVariables>& a, const Vector<NConditions>& b)
// {
// 	// init
// 	Vector<NConditions> basis;
// 	std::iota(basis.begin(), basis.end(), NVariables + 1);
// 	Vector<NVariables> free;
// 	std::iota(free.begin(), free.end(), 1);
// 	Matrix<NConditions + 1, NConditions + NVariables + 1> table = mathter::Zero();
// 	for (uint32_t i = 0; i != a.Height(); ++i)
// 	{
// 		for (uint32_t j = 0; j != a.Width(); ++j)
// 		{
// 			table(i, j) = a(i, j);
// 		}

// 	}
// 	for (uint32_t i = 0; i != NConditions; ++i)
// 	{
// 		table(i, i + NVariables) = 1;
// 	}
// 	for (uint32_t i = 0; i != NConditions; ++i)
// 	{
// 		table(i, NConditions + NVariables) = b[i];
// 	}
// 	for (uint32_t i = 0; i != NVariables; ++i)
// 	{
// 		table(NConditions, i) = -c[i];
// 	}
// 	std::cout << "basis : " << basis << std::endl;
// 	std::cout << "free  : " << free << std::endl;
// 	std::cout << table << std::endl;

// 	bool solution_is_find = false;

// 	while (!solution_is_find)
// 	{
// 		uint32_t curi = 0, curj = 0;
// 		for (; curj < NConditions + NVariables && table(NConditions, curj) >= 0; curj++);
// 		if (curj == NConditions + NVariables)
// 		{
// 			break;
// 		}
// 		value_type rat = INFINITY;
// 		for (int i = 0;i != NConditions; ++i)
// 		{
// 			if (abs(table(i, NConditions + NVariables) / table(i, curj)) < rat)
// 			{
// 				rat = abs(table(i, NConditions + NVariables) / table(i, curj));
// 				std::cout << rat << '\n';
// 				curi = i;
// 			}
// 		}
// 		std::swap(basis[curi], free[curj]);
// 		std::cout << "cur : " << curi << ' ' << curj << "(" << table(curi, curj) << ")" << std::endl;
// 		table(curi, curj) = 1 / table(curi, curj);
// 		for (int i = 0; i != NConditions + NVariables + 1; ++i)
// 		{
// 			if (i != curj)
// 			{
// 				table(curi, i) *= table(curi, curj);
// 			}
// 		}
// 		for (int i = 0; i != NConditions + 1; ++i)
// 		{
// 			if (i != curi)
// 			{
// 				table(i, curj) *= -table(curi, curj);
// 			}
// 		}
// 		for (size_t i = 0; i != NConditions + 1; i++)
// 		{
// 			for (int j = 0; j != NConditions + NVariables + 1; ++j)
// 			{
// 				if (i != curi && j != curj)
// 				{
// 					table(i, j) -= table(i, curj) * table(curi, j) / table(curi, curj);
// 				}
// 			}
// 		}

// 		std::cout << "basis : " << basis << std::endl;
// 		std::cout << "free  : " << free << std::endl;
// 		std::cout << table << std::endl;

// 		// return free;
// 	}
// 	return free;
// }

// int main(int argc, char const* argv[])
// {
// 	// (7, 0)
// 	// std::cout << " f(x,c) = -2x1 + 3x2;\n arg_min = {7, 0}, f(arg_min) =-14\n\n";
// 	Matrix<3, 2> A(-2, 6, 3, 2, 2, -1);
// 	Vector<2> C(2, 3);
// 	Vector<3> B(40, 28, 14);
// 	simplex_method(C, A, B);
// 	return 0;
// }
