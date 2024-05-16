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

void vprint(const vector_t* v)
{
	std::cout << "[";
	for (size_t i = 0; i < v->size - 1; ++i)
	{
		std::cout << vget(v, i) << "; ";
	}
	if (v->size != 0)
	{
		std::cout << vget(v, v->size - 1);
	}
	std::cout << "]";
}

void mprint(matrix_t* m)
{
	std::cout << "{";
	for (size_t i = 0; i < m->nrow - 1; ++i)
	{
		vector_t v;
		v.pdata = m->pdata + i * m->ncol;
		v.size = m->ncol;
		vprint(&v);
		std::cout << "\n";
	}
	if (m->nrow != 0)
	{
		vector_t v;
		v.pdata = m->pdata + (m->nrow - 1) * m->ncol;
		v.size = m->ncol;
		vprint(&v);
	}
	std::cout << "}\n";
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
	for (size_t i = 0; i != pdata->areaa.nrow; ++i)
	{
		mget(pout, i, i + pdata->areaa.ncol) = 1;
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
	const size_t cdepvar = ptable->ncol - pbasis->size;
	size_t curj = 0;
	size_t curi = 0;
	for (;curj != cdepvar && mget(ptable, ptable->nrow - 1, curj) >= 0; ++curj);
	if (curj == cdepvar)
	{
		return false;
	}
	for (size_t i = 0; i != pbasis->size; ++i)
	{
		if (mget(ptable, i, curj) < 0)
		{
			if (abs(mget(ptable, i, ptable->ncol - 1) / mget(ptable, i, curj)) <
				abs(mget(ptable, curi, ptable->ncol - 1) / mget(ptable, curi, curj)))
			{
				curi = i;
			}
		}
	}

	std::cout << "curi=" << curi << ", curj=" << curj << "\n";

	vget(pbasis, curi) = curj;
	num_t curval = mget(ptable, curi, curj);

	for (size_t j = 0; j != ptable->ncol;++j)
	{
		mget(ptable, curi, j) /= curval;
	}
	for (size_t i = 0; i != ptable->nrow; ++i)
	{
		if (i != curi)
		{
			num_t coeff = mget(ptable, i, curj);
			for (size_t j = 0; j != ptable->ncol; ++j)
			{
				mget(ptable, i, j) -= coeff * mget(ptable, curi, j);
			}
		}
	}
	return true;
}

void write_solution(const matrix_t* ptable, const vector_t* pbasis, lpsolution* pout)
{
	pout->point = vnew(ptable->ncol - pbasis->size);
	for (size_t i = 0; i != pout->point.size; ++i)
	{
		for (size_t j = 0; j != pbasis->size; ++j)
		{
			if (vget(pbasis, j) == i)
			{
				vget(&pout->point, i) = mget(ptable, j, ptable->ncol - 1);
			}
		}
	}
	pout->val = mget(ptable, ptable->nrow - 1, ptable->ncol - 1);
}

void lpsolve(const lpproblem* pdata, lpsolution* pout)
{
	matrix_t table;
	init_table(pdata, &table);
	std::cout << "INITIAL TABLE:\n";
	mprint(&table);
	std::cout << '\n';

	vector_t basis;
	init_basis(pdata, &basis);
	std::cout << "INITIAL BASIS:\n";
	vprint(&basis);
	std::cout << '\n';

	while (simplex_iteration(&table, &basis))
	{
		std::cout << "TABLE:\n";
		mprint(&table);
		std::cout << '\n';

		std::cout << "BASIS:\n";
		vprint(&basis);
		std::cout << '\n';
	}
	write_solution(&table, &basis, pout);
	delete[] table.pdata;
	delete[] basis.pdata;
}

int main(int argc, char const* argv[])
{
	matrix_t A = mnew(3, 2);
	A.pdata[0] = -2;
	A.pdata[1] = 6;
	A.pdata[2] = 3;
	A.pdata[3] = 2;
	A.pdata[4] = 2;
	A.pdata[5] = -1;

	vector_t C = vnew(2);
	C.pdata[0] = 2;
	C.pdata[1] = 3;

	vector_t B = vnew(3);
	B.pdata[0] = 40;
	B.pdata[1] = 28;
	B.pdata[2] = 14;

	lpproblem data;
	data.areaa = A;
	data.areab = B;
	data.funcc = C;

	lpsolution solution;

	lpsolve(&data, &solution);

	std::cout << "Minimum: " << solution.val << '\n';
	std::cout << "Minimal value: ";
	vprint(&solution.point);

	delete[] A.pdata;
	delete[] B.pdata;
	delete[] C.pdata;
	delete[] solution.point.pdata;
	return 0;
}


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
