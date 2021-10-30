/*
Вариат 15	
Б	
Строго случайные данные и случайные данные с малым числом уникальных значений
*/
#include <iostream>
#include <vector>
#include <iterator>

template< class RandomIt >
void sort(RandomIt first, RandomIt last)
{
    using std::swap;
    auto p = *(first + std::distance(first, last) / 2);
    auto i = first, j = last;
    while (i < j)
    {
        if (*i >= p)
        {
            while (*(--j) > p);
            if (i < j) swap(*i, *j);
        }
        ++i;
    }
    if (std::distance(first, last) > 2)
    {
        sort(first, i);
        sort(j, last);
    }
}

int main()
{
    std::vector<int> v = { 1, 3, 22, 456, 85, 234, 9, 56, 0, -22, 6 };
    sort(v.begin(), v.end());
    for (const auto& i : v)
        std::cout << i << ' ';
    std::cout << '\n';
}