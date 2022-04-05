#include <fstream>
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <cctype>
#include <iterator>

namespace{
    constexpr uint32_t ENTER_STATE = 1;
    constexpr char STACK_EMPTY_VAL = -1;
    constexpr size_t RU_ALPHABET_POWER = 33;
    constexpr size_t RU_PSM_STATES_NUM = 200'000;
}

class state;

template<size_t AlphaPower, size_t nStates>
using state_machine_t = std::array<std::array<state, nStates>, AlphaPower + 1>;

class state
{
    int64_t _id;
public:
    state() noexcept : _id(0) {}
    state(uint32_t state_id, bool available = false) noexcept : _id { state_id }
    {
        if (available) _id = -_id;
    }
    bool is_available() const noexcept { return _id >> 31; }
    void change_available() noexcept { _id = -_id; }
    void set_available() noexcept { _id = -std::abs(_id); }
    operator size_t() const noexcept { return std::abs(_id); }
    bool is_null() const noexcept { return !_id; }
};

class state_generator
{
    static uint32_t last_state;
public:
    static state next(bool available = false) noexcept { return { ++last_state, available }; }
    static void reset() noexcept { last_state = ENTER_STATE; }
    static uint32_t current() noexcept { return last_state; }
    static uint32_t peek() noexcept { return last_state + 1; }
    state_generator() = delete;
    state_generator(const state_generator&) = delete;
    state_generator(state_generator&&) = delete;
};

uint32_t state_generator::last_state = ENTER_STATE;

template<size_t AlphaPower, size_t nStates>
void fill_state_machine(state_machine_t<AlphaPower, nStates>& state_machine, uint16_t aWordSize, state enter, char* aFirstHalf) noexcept
{
    if (aWordSize)
        for (size_t i = 0; i != AlphaPower; ++i)
        {
            *aFirstHalf = i;
            if (state_machine[i][enter].is_null())
                state_machine[i][enter] = state_generator::next();
            fill_state_machine<AlphaPower, nStates>(state_machine, aWordSize - 1, state_machine[i][enter], aFirstHalf + 1);
        }
    else
    {
        state mid = state_generator::next();
        for (size_t i = 0; i != AlphaPower; ++i)
        {
            state_machine[i][enter] = mid;
            state_machine[i][mid] = state_generator::peek();
        }
        state cur_state = enter;
        while (*(--aFirstHalf) != STACK_EMPTY_VAL)
        {
            if (state_machine[*aFirstHalf][cur_state].is_null())
                state_machine[*aFirstHalf][cur_state] = state_generator::next();
            if (*(aFirstHalf - 1) == STACK_EMPTY_VAL)
                state_machine[*aFirstHalf][cur_state].set_available();
            cur_state = state_machine[*aFirstHalf][cur_state];
        }
    }
}

template<size_t AlphaPower, size_t nStates>
state_machine_t<AlphaPower, nStates>* create_state_machine(uint16_t aWordSize)
{
    auto palindrom_state_machine = new state_machine_t<AlphaPower, nStates>;
    char stack[6] = { STACK_EMPTY_VAL, 0, 0, 0, 0, 0 };
    for (uint16_t CurWordSize = aWordSize / 2; CurWordSize; --CurWordSize)
        fill_state_machine<AlphaPower, nStates>(*palindrom_state_machine, CurWordSize, ENTER_STATE, stack + 1);
    for (size_t i = 0; i != AlphaPower; ++i)
        (*palindrom_state_machine)[i][ENTER_STATE].set_available();
    std::cout << state_generator::current() << '\n';
    state_generator::reset();
    return palindrom_state_machine;
}

char* read_from_file(const char* filename)
{
    std::ifstream ifs { filename };
    size_t filesize = std::distance(std::istream_iterator<char>{ifs >> std::noskipws}, {});
    ifs.clear();
    ifs.seekg(std::ios::beg);
    char* filecontent = new char[filesize + 1];
    std::copy(std::istream_iterator<char>{ifs}, {}, filecontent);
    ifs.close();
    filecontent[filesize] = 0;
    return filecontent;
}

const char* next_lexem(const char* pos)
{
    while (std::isspace(*pos))
        ++pos;
    return pos;
}

const char* skip_lexem(const char* pos)
{
    while (*pos && !std::isspace(*pos))
        ++pos;
    return pos;
}

bool islexend(const char* pos)
{
    return std::isspace(*pos) || !(*pos);
}

size_t ru_to_index(char letter)
{
    // if ('А' <= letter && letter <= 'Я')
    //     return letter - 'A';
    // if ('а' <= letter && letter <= 'я')
    //     return letter - 'а';
    // if (letter == 'Ё' || letter == 'ё')
    //     return 32;
    return 33;
}

size_t en_to_index(char letter)
{
    return std::toupper(letter) - 'A';
}

std::vector<const char*> text_processing(const char* content)
{
    std::unique_ptr<state_machine_t<RU_ALPHABET_POWER, RU_PSM_STATES_NUM>> psm_table { create_state_machine<RU_ALPHABET_POWER, RU_PSM_STATES_NUM>(6) };
    std::vector<const char*> result;
    const char* curpos = content;
    while (*curpos)
    {
        const char* lexbeg = curpos;
        state curstate = ENTER_STATE;
        while (!curstate.is_null() && !islexend(curpos))
        {
            curstate = (*psm_table)[en_to_index(*curpos)][curstate];
            std::cout << curstate.is_available();
            ++curpos;
        }
        std::cout << '\n';
        if (curstate.is_available())
            result.emplace_back(lexbeg);
        if (curstate.is_null())
            curpos = skip_lexem(curpos);
        curpos = next_lexem(curpos);
    }
    return result;
}

void print(const std::vector<const char*>& lexems, const char* filename)
{
    std::ofstream ofs { filename };
    for (auto i = lexems.cbegin(); i != lexems.cend() - 1; ++i)
    {
        for (const char* cur = *i; !islexend(cur); ++cur)
            ofs << *cur;
        ofs << ' ';
    }
    ofs << *(lexems.cend() - 1);
    ofs.close();
}

int main()
{
    //setlocale(LC_ALL, "RU");
    char* content = read_from_file("input.txt");
    auto result = text_processing(content);
    print(result, "output.txt");
    delete[] content;
}