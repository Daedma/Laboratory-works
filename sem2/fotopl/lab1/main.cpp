#include <fstream>
#include <list>
#include <cctype>
#include <algorithm>
#include <memory>
#include <iterator>

template<typename T>
void realloc(T*& old, size_t old_size, size_t new_size)
{
    T* new_p = new T[new_size];
    std::copy_n(old, old_size, new_p);
    delete[] old;
    old = new_p;
}

char* read_word(std::ifstream& afile)
{
    char cur_ch;
    char* word = new char[10];
    size_t word_size = 0;
    size_t word_capacity = 10;
    for (; afile.get(cur_ch) && !std::isspace(cur_ch); ++word_size)
    {
        word[word_size] = cur_ch;
        if (word_size == word_capacity - 1)
        {
            realloc(word, word_capacity, word_capacity * 2);
            word_capacity *= 2;
        }
    }
    if (word_size == 0)
        return nullptr;
    word[word_size] = 0;
    return word;
}

bool task(char* str)
{
    /*
    Обработка...
    */
    return true;
}

int main()
{
    std::ifstream ifs { "input.txt" };
    char* cur;
    std::list<std::unique_ptr<char[]>> words;
    while ((cur = read_word(ifs)))
        words.emplace_back(cur);
    ifs.close();
    words.remove_if([](const auto& val){return !task(val.get()); });
    std::ofstream ofs { "output.txt" };
    std::transform(words.cbegin(), words.cend(), std::ostream_iterator<char*>{ofs}, [](const auto& val){return val.get()});
    ofs.close();
}