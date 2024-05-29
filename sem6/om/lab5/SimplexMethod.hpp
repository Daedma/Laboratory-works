#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <numeric>
#include <map>

template<typename T>
class SimplexMethod
{
public:
	SimplexMethod(const std::vector<std::vector<T>>& A,
		const std::vector<T>& b,
		const std::vector<T>& c);

	bool solve();

	const std::vector<T>& getSolution() const
	{
		return solution;
	}
	const T getObjectiveValue() const
	{
		return objectiveValue;
	}

private:
	void printTable() const;
	void pivot(int pivotRow, int pivotCol);
	int findPivotRow(int pivotCol);
	int findPivotCol();
	bool isOptimal();

	std::vector<std::vector<T>> tableau;
	std::vector<T> solution;
	T objectiveValue;
	std::vector<size_t> varMapping;
	size_t numIterations;
	size_t m;
	size_t n;
};

template<typename T>
SimplexMethod<T>::SimplexMethod(const std::vector<std::vector<T>>& A,
	const std::vector<T>& b,
	const std::vector<T>& c) : m(A.size()), n(A[0].size())
{
	tableau.resize(m + 1, std::vector<T>(n + m + 1, (T(0))));

	for (size_t i = 0; i != m; ++i)
	{
		for (size_t j = 0; j != n; ++j)
		{
			tableau[i][j] = A[i][j];
		}
		tableau[i][n + i] = 1;
		tableau[i][n + m] = b[i];
	}
	for (size_t j = 0; j != n; ++j)
	{
		tableau[m][j] = -c[j];
	}
	tableau[m][n + m] = 0;

	solution.resize(n);

	varMapping.resize(n);
	std::iota(varMapping.begin(), varMapping.end(), T{ 0 });

	numIterations = 0;
}

template<typename T>
bool SimplexMethod<T>::solve()
{
	using std::swap;
	printTable();
	while (!isOptimal())
	{
		int pivotCol = findPivotCol();
		if (pivotCol == -1)
		{
			return false; // No solution
		}
		int pivotRow = findPivotRow(pivotCol);
		if (pivotRow == -1)
		{
			return false; // Unbounded solution
		}
		pivot(pivotRow, pivotCol);
		varMapping[pivotCol] = pivotRow;
		++numIterations;
		printTable();
	}

	for (size_t i = 0; i != n; ++i)
	{
		solution[i] = tableau[varMapping[i]][n + m];
	}
	objectiveValue = -tableau[m][n + m];

	return true;
}

template<typename T>
void SimplexMethod<T>::pivot(int pivotRow, int pivotCol)
{
	T pivotElement = tableau[pivotRow][pivotCol];

	for (size_t j = 0; j != n + m + 1; ++j)
	{
		tableau[pivotRow][j] /= pivotElement;
	}

	for (size_t i = 0; i != m + 1; ++i)
	{
		if (i != pivotRow)
		{
			T factor = tableau[i][pivotCol];
			for (size_t j = 0; j != n + m + 1; ++j)
			{
				tableau[i][j] -= factor * tableau[pivotRow][j];
			}
		}
	}
}

template<typename T>
int SimplexMethod<T>::findPivotRow(int pivotCol)
{
	T minRatio = std::numeric_limits<T>::max();
	int pivotRow = -1;

	for (size_t i = 0; i != tableau.size(); ++i)
	{
		if (tableau[i][pivotCol] > 0)
		{
			T ratio = tableau[i][tableau[0].size() - 1] / tableau[i][pivotCol];
			if (ratio < minRatio)
			{
				minRatio = ratio;
				pivotRow = i;
			}
		}
	}

	return pivotRow;
}

template<typename T>
int SimplexMethod<T>::findPivotCol()
{
	T maxCoeff = std::numeric_limits<T>::lowest();
	int pivotCol = -1;

	for (size_t j = 0; j != tableau[0].size() - 1; ++j)
	{
		if (tableau[tableau.size() - 1][j] < 0)
		{
			T coeff = -tableau[tableau.size() - 1][j];
			if (coeff > maxCoeff)
			{
				maxCoeff = coeff;
				pivotCol = j;
			}
		}
	}

	return pivotCol;
}

template<typename T>
bool SimplexMethod<T>::isOptimal()
{
	for (size_t j = 0; j != tableau[0].size() - 1; ++j)
	{
		if (tableau[tableau.size() - 1][j] < 0)
		{
			return false;
		}
	}
	return true;
}

template<typename T>
void SimplexMethod<T>::printTable() const
{
	for (size_t i = 0; i != m + 1; ++i)
	{
		for (size_t j = 0; j != n + m + 1; ++j)
		{
			std::cout << std::setw(10) << tableau[i][j];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
