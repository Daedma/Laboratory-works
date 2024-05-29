#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <numeric>

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
	size_t numIterations;
	size_t m;
	size_t n;
};

template<typename T>
SimplexMethod<T>::SimplexMethod(const std::vector<std::vector<T>>& A,
	const std::vector<T>& b,
	const std::vector<T>& c) : m(A.size()), n(A[0].size())
{
	// Initialize tableau
	tableau.resize(m + 1, std::vector<T>(n + m + 1));

	// Fill tableau with A, b, and c
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			tableau[i][j] = A[i][j];
		}
		tableau[i][n + i] = 1;
		tableau[i][n + m] = b[i];
	}
	for (int j = 0; j < n; ++j)
	{
		tableau[m][j] = -c[j];
	}
	tableau[m][n + m] = 0;

	// Initialize other variables
	solution.resize(n);
	numIterations = 0;
}

template<typename T>
bool SimplexMethod<T>::solve()
{
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
		++numIterations;
		printTable();
	}

	for (int j = 0; j < n; ++j)
	{
		solution[j] = tableau[j][n + m];
	}
	objectiveValue = -tableau[m][n + m];

	return true;
}

template<typename T>
void SimplexMethod<T>::pivot(int pivotRow, int pivotCol)
{
	T pivotElement = tableau[pivotRow][pivotCol];

	// Divide the pivot row by the pivot element
	for (int j = 0; j <= n + m; ++j)
	{
		tableau[pivotRow][j] /= pivotElement;
	}

	// Perform row operations to eliminate the pivot column
	for (int i = 0; i < m + 1; ++i)
	{
		if (i == pivotRow)
		{
			continue;
		}
		T factor = tableau[i][pivotCol];
		for (int j = 0; j <= n + m; ++j)
		{
			tableau[i][j] -= factor * tableau[pivotRow][j];
		}
	}
}

template<typename T>
int SimplexMethod<T>::findPivotRow(int pivotCol)
{
	T minRatio = std::numeric_limits<T>::max();
	int pivotRow = -1;

	for (int i = 0; i < tableau.size(); ++i)
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

	for (int j = 0; j != tableau[0].size() - 1; ++j)
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
	for (int i = 0; i < m + 1; ++i)
	{
		for (int j = 0; j < n + m + 1; ++j)
		{
			std::cout << std::setw(10) << tableau[i][j];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// template<typename T>
// void SimplexMethod<T>::printTable() const
// {
// 	std::cout << "Iteration " << numIterations << ":" << std::endl;

// 	// Определяем максимальную ширину столбцов
// 	int maxWidth = 0;
// 	for (size_t j = 0; j < tableau[0].size(); ++j)
// 	{
// 		int width = 0;
// 		for (size_t i = 0; i < tableau.size(); ++i)
// 		{
// 			width = std::max(width, static_cast<int>(std::floor(std::log10(std::abs(tableau[i][j]))) + 1));
// 		}
// 		maxWidth = std::max(maxWidth, width);
// 	}

// 	// Выводим заголовки таблицы
// 	std::cout << std::left << std::setw(maxWidth + 1) << "  ";
// 	for (size_t j = 0; j < tableau[0].size(); ++j)
// 	{
// 		if (j < m && tableau[0][j] == j + 1)
// 		{
// 			std::cout << std::left << std::setw(maxWidth + 1) << "x" << j + 1;
// 		}
// 		else if (j == tableau[0].size() - 1)
// 		{
// 			std::cout << std::left << std::setw(maxWidth + 1) << "b";
// 		}
// 		else
// 		{
// 			std::cout << std::left << std::setw(maxWidth + 1) << "";
// 		}
// 	}
// 	std::cout << std::endl;

// 	// Выводим строки таблицы
// 	for (size_t i = 0; i < tableau.size(); ++i)
// 	{
// 		if (i < m && tableau[i][tableau[0].size() - 1] == 1)
// 		{
// 			std::cout << std::left << std::setw(maxWidth + 1) << "x" << i + 1 << ":";
// 		}
// 		else if (i == tableau.size() - 1)
// 		{
// 			std::cout << std::left << std::setw(maxWidth + 1) << "f:";
// 		}
// 		else
// 		{
// 			std::cout << std::left << std::setw(maxWidth + 1) << "";
// 		}
// 		for (size_t j = 0; j < tableau[i].size(); ++j)
// 		{
// 			std::cout << std::right << std::setw(maxWidth + 1) << tableau[i][j];
// 		}
// 		std::cout << std::endl;
// 	}

// 	// Выводим значения базисных переменных
// 	std::cout << "Basic variables:" << std::endl;
// 	for (size_t i = 0; i < m; ++i)
// 	{
// 		if (tableau[i][tableau[0].size() - 1] == 1)
// 		{
// 			std::cout << std::left << std::setw(maxWidth + 1) << "x" << i + 1 << ": " << solution[i] << std::endl;
// 		}
// 	}
// 	std::cout << std::left << std::setw(maxWidth + 1) << "f: " << -objectiveValues.back() << std::endl;
// }


