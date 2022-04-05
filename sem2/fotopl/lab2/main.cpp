#include <fstream>
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <string_view>
#include <cctype>
#include <iterator>

namespace{
    constexpr uint32_t ENTER_STATE = 1;
    constexpr char STACK_EMPTY_VAL = -1;
    constexpr size_t RU_ALPHABET_POWER = 66;
    constexpr size_t RU_PSM_STATES_NUM = 291920;
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

state next_state(bool available = false) noexcept
{
    static uint32_t last_state = ENTER_STATE;
    return { ++last_state, available };
}

template<size_t AlphaPower, size_t nStates>
void fill_state_machine(state_machine_t<AlphaPower, nStates>& state_machine, uint16_t aWordSize, state enter, char* aFirstHalf) noexcept
{
    if (aWordSize)
        for (size_t i = 0; i != AlphaPower; ++i)
        {
            *aFirstHalf = i;
            if (state_machine[i][enter].is_null())
                state_machine[i][enter] = next_state();
            fill_state_machine<AlphaPower, nStates>(state_machine, aWordSize - 1, state_machine[i][enter], aFirstHalf + 1);
        }
    else
    {
        for (size_t i = 0; i != AlphaPower; ++i)
        {
            state_machine[i][enter] = enter;
        }
        state cur_state = enter;
        while (*(--aFirstHalf) != STACK_EMPTY_VAL)
        {
            if (state_machine[*aFirstHalf][cur_state].is_null())
                state_machine[*aFirstHalf][cur_state] = next_state();
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
    return palindrom_state_machine;
}

size_t file_char_count(std::ifstream& ifs)
{
    return std::distance(std::istream_iterator<char>{ifs >> std::noskipws}, {});
}

char* read_from_file(const char* filename)
{
    std::ifstream ifs { filename };
    size_t filesize = file_char_count(ifs);
    ifs.close();
    ifs.open(filename);
    char* filecontent = new char[filesize + 1];
    std::copy(std::istream_iterator<char>{ifs}, {}, filecontent);
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
    return std::isspace(*pos);
}

size_t ru_to_index(char letter)
{
    // if ('А' <= letter && letter <= 'я')
    //     return letter + 'А';
    // if (letter == 'Ё')
    //     return 64;
    // if (letter == 'ё')
    //     return 65;
    return 66;
}

size_t en_to_index(char letter)
{
    return letter - 'A';
}

std::vector<std::string_view> text_processing(const char* content)
{
    std::unique_ptr<state_machine_t<RU_ALPHABET_POWER, RU_PSM_STATES_NUM>> psm_table { create_state_machine<RU_ALPHABET_POWER, RU_PSM_STATES_NUM>(6) };
    std::vector<std::string_view> result;
    const char* curpos = content;
    while (*curpos)
    {
        const char* lexbeg = curpos;
        size_t lexsize = 0;
        state curstate = ENTER_STATE;
        while (!curstate.is_null() && !islexend(curpos) && *curpos)
        {
            curstate = (*psm_table)[en_to_index(*curpos)][curstate];
            ++curpos;
            ++lexsize;
        }
        if (curstate.is_available())
        {
            result.emplace_back(lexbeg, lexsize);
        }
        else if (curstate.is_null())
            curpos = skip_lexem(curpos);
        curpos = next_lexem(curpos);
    }
    return result;
}

int main()
{
    //setlocale(LC_ALL, "RU");
    char* content = read_from_file("input.txt");
    auto result = text_processing(content);
    std::ofstream ofs { "output.txt" };
    std::copy(result.cbegin(), result.cend(), std::ostream_iterator<std::string_view>{ofs, " "});
    std::cout << next_state();
    delete[] content;
}