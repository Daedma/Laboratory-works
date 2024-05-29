// ЛР 5: Симплекс метод прямой и двухэтапный
#include <iostream>
#include <vector>
#include "SimplexMethod.hpp"  // Подключаем заголовочный файл с классом SimplexMethod

int main()
{
	// std::cout << " f(x,c) = -2x1 + 3x2;\n arg_min = {7, 0}, f(arg_min) =-14\n\n";
// Создаем матрицу ограничений A и векторы C и B
	std::vector<std::vector<double>> A = { { -2, 6 }, { 3, 2 }, { 2, -1 } };
	std::vector<double> C = { 2, 3 };
	std::vector<double> B = { 40, 28, 14 };

	// Создаем экземпляр класса SimplexMethod и вызываем метод solve
	SimplexMethod<double> simplex(A, B, C);
	if (simplex.solve())
	{
// Выводим результаты, если решение найдено
		std::cout << "Optimal solution found:" << std::endl;
		std::cout << "Objective value: " << -simplex.getObjectiveValue() << std::endl;
		std::cout << "Variables:" << std::endl;
		for (size_t i = 0; i < simplex.getSolution().size(); ++i)
		{
			std::cout << "x" << i + 1 << ": " << simplex.getSolution()[i] << std::endl;
		}
	}
	else
	{
	 // Выводим сообщение об ошибке, если решение не найдено
		std::cout << "No optimal solution found." << std::endl;
	}

	return 0;
}
