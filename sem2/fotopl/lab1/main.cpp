#include <fstream>
#include <vector>
#include <cctype>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <cstring>
#include <iostream>

//перевыделение памяти под массив
template <typename T>
void realloc(T *&old, size_t old_size, size_t new_size)
{
    T *new_p = new T[new_size];
    std::copy_n(old, old_size, new_p);
    delete[] old;
    old = new_p;
}

//считать содержимое файла в строку
char *read(std::ifstream &afile)
{
    char *text = new char[10];
    size_t text_size = 0;
    size_t text_capacity = 10;
    for (char cur_ch; afile.get(cur_ch); ++text_size)
    {
        text[text_size] = cur_ch;
        if (text_size == text_capacity - 1)
        {
            realloc(text, text_capacity, text_capacity * 2);
            text_capacity *= 2;
        }
    }
    if (text_size == 0)
    {
        delete[] text;
        return nullptr;
    }
    text[text_size] = 0;
    return text;
}

//проверка слова на палиндром
bool is_palindrom(char *word)
{
    for (char *end = word + std::strlen(word) - 1; word < end; ++word, --end)
        if (*word != *end)
            return false;
    return true;
}

//проверка слова на русский язык
bool is_ru(char *word)
{
    while (*word)
    {
        if (!(*word >= 'А' || *word == 'Ё' || *word == 'ё'))
            return false;
        ++word;
    }
    return true;
}

//начало следующего слова
char *next_word_begin(char *p)
{
    while (std::isspace(*p))
        ++p;
    return p;
}

//конец текущего слова
char *cur_word_end(char *p)
{
    while (!std::isspace(*p) && *p)
        ++p;
    return p;
}

//разделяет текст на слова
std::vector<char *> split(char *text)
{
    std::vector<char *> words;
    for (char *word_beg = next_word_begin(text), *word_end = cur_word_end(word_beg); word_beg != word_end; word_beg = next_word_begin(word_end), word_end = cur_word_end(word_beg))
    {
        char *cur_word = new char[word_end - word_beg + 1];
        cur_word[word_end - word_beg] = 0;
        std::copy(word_beg, word_end, cur_word);
        words.emplace_back(cur_word);
    }
    return words;
}
//возвращает вектор слов, удовлетворяющих условию задания
std::vector<char *> task(char *text)
{
    std::vector<char *> results;
    auto words = split(text);
    std::copy_if(words.cbegin(), words.cend(), std::inserter(results, results.end()),
                 [](char *word)
                 {
                     if (std::strlen(word) <= 6 && is_ru(word) && is_palindrom(word))
                         return true;
                     delete[] word;
                     return false;
                 });

    return results;
}

int main()
{
    setlocale(LC_ALL, "RU");
    std::ifstream ifs{"input.txt"};
    char *text = read(ifs);
    ifs.close();
    auto results = task(text); //слова, удовлетворяющие условию задания
    delete[] text;
    std::copy(results.cbegin(), results.cend(), std::ostream_iterator<char *>{std::cout, " "});
    std::ofstream ofs{"output.txt"};
    std::copy(results.cbegin(), results.cend(), std::ostream_iterator<char *>{ofs, " "});
    ofs.close();
    for (char *i : results)
        delete[] i;
}