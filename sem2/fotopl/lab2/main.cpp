#include <fstream>
#include <array>
//#include <cmath>
//#include <stack>
#include <iostream>

#define SIGNBIT_MASK 0x80000000ULL
#define STACK_EMPTY_VAL -1

constexpr std::array<char, 2> alphabet = { 'a', 'b' };

template<typename T>
inline constexpr T absolute(T val) noexcept
{
    return val < 0 ? -val : val;
}

class state
{
    int64_t _id;
public:
    constexpr state() noexcept : _id(0) {}
    constexpr state(uint32_t state_id, bool available = false) noexcept : _id { state_id }
    {
        if (available) _id = -_id;
    }
    constexpr bool is_available() const noexcept { return _id >> 31; }
    constexpr void change_available() noexcept { _id = -_id; }
    constexpr void set_available() noexcept { _id = -absolute(_id); }
    constexpr operator size_t() const noexcept { return absolute(_id); }
    constexpr bool is_null() const noexcept { return !_id; }
};

state next_state(bool available = false) noexcept
{
    static uint32_t last_state = 0;
    return { ++last_state, available };
}

constexpr size_t calc_states_num(size_t aAplha_power) noexcept
{
    return 64;
}

template<typename T>
constexpr inline bool is_odd(T val) noexcept { return val & 1; }

template<size_t S>
void fill_state_machine(std::array < std::array < state, calc_states_num(S)>, S>& state_machine, uint16_t aWordSize, state enter, char* aFirstHalf) noexcept
{
    if (aWordSize)
        for (char i = 0; i != S; ++i)
        {
            *aFirstHalf = i;
            if (state_machine[i][enter].is_null())
                state_machine[i][enter] = next_state();
            if (aWordSize == 1)
            {
                for (char j = 0; j != S; ++j)
                {
                    state_machine[j][enter] = state_machine[i][enter];
                }
            }
            fill_state_machine(state_machine, aWordSize - 1, state_machine[i][enter], aFirstHalf + 1);
        }
    else
    {
        state cur_state = enter;
        while (*(--aFirstHalf) != STACK_EMPTY_VAL)
        {
            if (state_machine[*aFirstHalf][cur_state].is_null())
                state_machine[*aFirstHalf][cur_state] = next_state();
            cur_state = state_machine[*aFirstHalf][cur_state];
        }
        state_machine[*(aFirstHalf + 1)][cur_state].set_available();
    }
}

template<size_t S>
std::array < std::array < state, calc_states_num(S)>, S> create_state_machine(const std::array<char, S>& aAlphabet, uint16_t aWordSize)
{
    std::array < std::array < state, calc_states_num(S)>, S> palindrom_state_machine;
    state init_state = next_state();
    char stack[6] = { STACK_EMPTY_VAL, 0, 0, 0, 0, 0 };
    char* iter = stack + 1;
    for (uint16_t CurWordSize = aWordSize / 2; CurWordSize; --CurWordSize)
    {
        fill_state_machine(palindrom_state_machine, CurWordSize, init_state, iter);
    }
    return palindrom_state_machine;
}

bool check_word(const char* word)
{
    auto table = create_state_machine(alphabet, 6);
    std::ofstream ofs { "table.csv" };
    for (int i = 0; i != 80; ++i)
    {
        ofs << i << ", ";
    }
    ofs << '\n';
    for (auto i : table)
    {
        for (auto j : i)
            ofs << j << ", ";
        ofs << '\n';
    }
    state cur_state = 1;
    while (*word)
    {
        cur_state = table[*word++ - 97][cur_state];
        std::cout << cur_state.is_available() << ' ';
    }
    return cur_state.is_available();
}

int main()
{
    try
    {
        std::cout << check_word("bbb");
    }
    catch (std::runtime_error e)
    {
        std::cerr << e.what();
    }
}