#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>

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
	const std::vector<T>& getObjectiveValues() const
	{
		return objectiveValues;
	}

private:
	void printTable() const;
	void pivot(int pivotRow, int pivotCol);
	int findPivotRow(int pivotCol);
	int findPivotCol();
	void addArtificialVariables();
	void removeArtificialVariables();
	bool isOptimal();

	std::vector<std::vector<T>> tableau;
	std::vector<T> solution;
	std::vector<T> objectiveValues;
	std::vector<int> artificialVariables;
	int numIterations;
	int m;
	int n;
};

template<typename T>
SimplexMethod<T>::SimplexMethod(const std::vector<std::vector<T>>& A,
	const std::vector<T>& b,
	const std::vector<T>& c)
{
// Проверяем, что размеры векторов соответствуют требуемым
	if (A.empty() || b.empty() || c.empty() || A.size() != b.size() || A[0].size() != c.size())
	{
		throw std::invalid_argument("Invalid sizes of input vectors.");
	}

	m = A.size();
	n = A[0].size();

	// Формируем исходную симплекс-таблицу
	tableau.resize(m + 1, std::vector<T>(n + m + 1, (T(0))));

	// Заполняем таблицу ограничениями
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			tableau[i][j] = A[i][j];
		}
		tableau[i][n + i] = 1;
		tableau[i][n + m] = b[i];
	}

	// Заполняем таблицу целевой функцией
	for (int j = 0; j < n; ++j)
	{
		tableau[m][j] = -c[j];
	}
	tableau[m][n + m] = 0;

	// Инициализируем вектора решения и значений целевой функции
	solution.resize(n);
	objectiveValues.push_back(tableau[m][n + m]);

	// Инициализируем счетчик итераций
	numIterations = 0;
	printTable();
}

template<typename T>
bool SimplexMethod<T>::solve()
{
	while (true)
	{
// Добавляем искусственные переменные, если необходимо
		if (tableau[0][tableau.size() - 1] < 0)
		{
			addArtificialVariables();
		}

		// Проверяем, является ли текущее решение оптимальным
		if (isOptimal())
		{
			removeArtificialVariables();
			return true;
		}

		// Выбираем опорный столбец
		int pivotCol = findPivotCol();
		if (pivotCol == -1)
		{
// Решение не существует или неограниченно
			removeArtificialVariables();
			return false;
		}

		// Выбираем опорную строку
		int pivotRow = findPivotRow(pivotCol);

		// Выполняем пересчет таблицы
		pivot(pivotRow, pivotCol);

		// Обновляем вектор решения и значение целевой функции
		for (int j = 0; j < tableau[0].size(); ++j)
		{
			if (tableau[pivotRow][j] != 0)
			{
				solution[tableau[0][j] - 1] = tableau[pivotRow][tableau[0].size() - 1] / tableau[pivotRow][j];
			}
			else
			{
				solution[tableau[0][j] - 1] = 0;
			}
		}
		objectiveValues.push_back(tableau[tableau.size() - 1][tableau[0].size() - 1]);

		// Увеличиваем счетчик итераций
		numIterations++;
		printTable();
	}
}

template<typename T>
void SimplexMethod<T>::pivot(int pivotRow, int pivotCol)
{
	T pivotElement = tableau[pivotRow][pivotCol];

	// Делим опорную строку на опорный элемент
	for (int j = 0; j < tableau[0].size(); ++j)
	{
		tableau[pivotRow][j] /= pivotElement;
	}

	// Вычитаем опорную строку из всех остальных строк, умноженную на соответствующий элемент
	for (int i = 0; i < tableau.size(); ++i)
	{
		if (i != pivotRow)
		{
			T multiplier = tableau[i][pivotCol];
			for (int j = 0; j < tableau[0].size(); ++j)
			{
				tableau[i][j] -= multiplier * tableau[pivotRow][j];
			}
		}
	}

	// Обновляем индексы базисных переменных
	std::swap(tableau[0][pivotCol], tableau[0][tableau[0].size() - n]);
	std::swap(tableau[pivotRow][tableau[0].size() - n], tableau[pivotRow][tableau[0].size() - 1]);
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
		if (tableau[tableau.size() - 1][j] < 0 && tableau[0][j] >= 0)
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
void SimplexMethod<T>::addArtificialVariables()
{
	int m = tableau.size() - 1;
	int n = tableau[0].size() - m - 1;

	// Добавляем искусственные переменные в таблицу
	for (int i = 0; i < m; ++i)
	{
		tableau[i][tableau[0].size() - n - 1 - i] = 1;
		tableau[tableau.size() - 1][tableau[0].size() - n - 1 - i] = 1;
	}

	// Обновляем индексы базисных переменных
	for (int i = 0; i < m; ++i)
	{
		std::swap(tableau[0][tableau[0].size() - n - 1 - i], tableau[0][n + i]);
	}

	// Сохраняем индексы искусственных переменных
	artificialVariables.resize(m);
	for (int i = 0; i < m; ++i)
	{
		artificialVariables[i] = tableau[0].size() - n - 1 - i;
	}
}

template<typename T>
void SimplexMethod<T>::removeArtificialVariables()
{
// Удаляем столбцы искусственных переменных из таблицы
	for (int i = 0; i < artificialVariables.size(); ++i)
	{
		for (int j = 0; j < tableau.size(); ++j)
		{
			tableau[j].erase(tableau[j].begin() + artificialVariables[i]);
		}
		tableau[0].erase(tableau[0].begin() + artificialVariables[i]);
	}

	// Обновляем размеры таблицы
	for (int i = 0; i < tableau.size(); ++i)
	{
		tableau[i].resize(tableau[0].size());
	}

	// Очищаем вектор индексов искусственных переменных
	artificialVariables.clear();
}

template<typename T>
bool SimplexMethod<T>::isOptimal()
{
	for (int j = 0; j < tableau[0].size() - 1; ++j)
	{
		if (tableau[tableau.size() - 1][j] < 0)
		{
			return false;
		}
	}

	// Проверяем, что все искусственные переменные равны нулю
	for (int i = 0; i < artificialVariables.size(); ++i)
	{
		if (tableau[i][tableau[0].size() - 1] > 0)
		{
			return false;
		}
	}

	return true;
}

template<typename T>
void SimplexMethod<T>::printTable() const
{
	std::cout << "Iteration " << numIterations << ":" << std::endl;

	// Выводим заголовки таблицы
	std::cout << "  ";
	for (size_t j = 0; j < tableau[0].size(); ++j)
	{
		if (j < m && tableau[0][j] == j + 1)
		{
			std::cout << "x" << j + 1 << " ";
		}
		else if (j == tableau[0].size() - 1)
		{
			std::cout << "b  ";
		}
		else
		{
			std::cout << "   ";
		}
	}
	std::cout << std::endl;

	// Выводим строки таблицы
	for (size_t i = 0; i < tableau.size(); ++i)
	{
		if (i < m && tableau[i][tableau[0].size() - 1] == 1)
		{
			std::cout << "x" << i + 1 << ": ";
		}
		else if (i == tableau.size() - 1)
		{
			std::cout << "f: ";
		}
		else
		{
			std::cout << "  ";
		}
		for (size_t j = 0; j < tableau[i].size(); ++j)
		{
			std::cout << tableau[i][j] << " ";
		}
		std::cout << std::endl;
	}

	// Выводим значения базисных переменных
	std::cout << "Basic variables:" << std::endl;
	for (size_t i = 0; i < m; ++i)
	{
		if (tableau[i][tableau[0].size() - 1] == 1)
		{
			std::cout << "x" << i + 1 << ": " << solution[i] << std::endl;
		}
	}
	std::cout << "f: " << -objectiveValues.back() << std::endl;
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


