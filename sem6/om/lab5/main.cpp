// ЛР 5: Симплекс метод прямой и двухэтапный

#include <iostream>
#include <functional>
#include <cmath>
#include <utility>
#include <numeric>
#include "Mathter/Vector.hpp"
#include "Mathter/Matrix.hpp"
#include "Mathter/IoStream.hpp"

using value_type = double;

template<int N>
using Vector = mathter::Vector<value_type, N>;

template<int Rows, int Columns>
using Matrix = mathter::Matrix<value_type, Rows, Columns, mathter::eMatrixOrder::PRECEDE_VECTOR>;

template<int NVariables, int NConditions>
Vector<NVariables> simplex_method(const Vector<NVariables>& c, const Matrix<NConditions, NVariables>& a, const Vector<NConditions>& b)
{
	// init
	Vector<NConditions> basis;
	std::iota(basis.begin(), basis.end(), NVariables + 1);
	Vector<NVariables> free;
	std::iota(free.begin(), free.end(), 1);
	Matrix<NConditions + 1, NConditions + NVariables + 1> table = mathter::Zero();
	for (uint32_t i = 0; i != a.Height(); ++i)
	{
		for (uint32_t j = 0; j != a.Width(); ++j)
		{
			table(i, j) = a(i, j);
		}

	}
	for (uint32_t i = 0; i != NConditions; ++i)
	{
		table(i, i + NVariables) = 1;
	}
	for (uint32_t i = 0; i != NConditions; ++i)
	{
		table(i, NConditions + NVariables) = b[i];
	}
	for (uint32_t i = 0; i != NVariables; ++i)
	{
		table(NConditions, i) = -c[i];
	}
	std::cout << "basis : " << basis << std::endl;
	std::cout << "free  : " << free << std::endl;
	std::cout << table << std::endl;

	bool solution_is_find = false;

	while (!solution_is_find)
	{
		uint32_t curi = 0, curj = 0;
		for (; curj < NConditions + NVariables && table(NConditions, curj) >= 0; curj++);
		if (curj == NConditions + NVariables)
		{
			break;
		}
		value_type rat = INFINITY;
		for (int i = 0;i != NConditions; ++i)
		{
			if (abs(table(i, NConditions + NVariables) / table(i, curj)) < rat)
			{
				rat = abs(table(i, NConditions + NVariables) / table(i, curj));
				std::cout << rat << '\n';
				curi = i;
			}
		}
		std::swap(basis[curi], free[curj]);
		std::cout << "cur : " << curi << ' ' << curj << "(" << table(curi, curj) << ")" << std::endl;
		table(curi, curj) = 1 / table(curi, curj);
		for (int i = 0; i != NConditions + NVariables + 1; ++i)
		{
			if (i != curj)
			{
				table(curi, i) *= table(curi, curj);
			}
		}
		for (int i = 0; i != NConditions + 1; ++i)
		{
			if (i != curi)
			{
				table(i, curj) *= -table(curi, curj);
			}
		}
		for (size_t i = 0; i != NConditions + 1; i++)
		{
			for (int j = 0; j != NConditions + NVariables + 1; ++j)
			{
				if (i != curi && j != curj)
				{
					table(i, j) -= table(i, curj) * table(curi, j) / table(curi, curj);
				}
			}
		}

		std::cout << "basis : " << basis << std::endl;
		std::cout << "free  : " << free << std::endl;
		std::cout << table << std::endl;

		// return free;
	}
	return free;
}

int main(int argc, char const* argv[])
{
	// (7, 0)
	// std::cout << " f(x,c) = -2x1 + 3x2;\n arg_min = {7, 0}, f(arg_min) =-14\n\n";
	Matrix<3, 2> A(-2, 6, 3, 2, 2, -1);
	Vector<2> C(2, 3);
	Vector<3> B(40, 28, 14);
	simplex_method(C, A, B);
	return 0;
}
