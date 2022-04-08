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
#include <iostream>

namespace{
    constexpr uint32_t ENTER_STATE = 1;//Входное состояние
    constexpr char STACK_EMPTY_VAL = -1;//Специальное значение для обнаружения начала стека
    constexpr size_t RU_ALPHABET_POWER = 33;//Мощность русского алфавита
    constexpr size_t RU_PSM_STATES_NUM = 145928;//Количество генерируемых состояний//37'061
}

class state;

template<size_t AlphaPower, size_t nStates>
using state_machine_t = std::array<std::array<state, nStates>, AlphaPower + 1>;//Тип таблицы, задающей автомат

class state//Класс состояния
{
    int32_t _id;
public:
    state() noexcept : _id(0) {}
    state(uint32_t state_id, bool available = false) noexcept : _id { static_cast<int32_t>(state_id) }
    {
        if (available) _id = -_id;
    }
    bool is_available() const noexcept { return _id >> 31; }//Проверка на допустимость состояния
    void set_available() noexcept { _id = -std::abs(_id); }//Сделать состояние допустимым
    operator size_t() const noexcept { return std::abs(_id); }//Вернуть номер состояния
    bool is_null() const noexcept { return !_id; }//Проверка на нулевое состояние, из которого нельзя выбраться
};

class state_generator//Генератор состояний
{
    static uint32_t last_state;//Последнее сгенерированное состояние
public:
    static state next(bool available = false) noexcept { return { ++last_state, available }; }//Получить следующее состояние
    static void reset() noexcept { last_state = ENTER_STATE; }//Сбросить генератор к начальному значению
    static uint32_t current() noexcept { return last_state; }//Посмотреть номер последнего сгенерированного состояния
    static uint32_t peek() noexcept { return last_state + 1; }//Посмотреть следующее состояние, которое будет сгенерированно
    state_generator() = delete;
    state_generator(const state_generator&) = delete;
    state_generator(state_generator&&) = delete;
};

uint32_t state_generator::last_state = ENTER_STATE;

template<size_t AlphaPower, size_t nStates>
void fill_state_machine(state_machine_t<AlphaPower, nStates>& state_machine, uint16_t aWordSize, state enter, char* aFirstHalf) noexcept//Функция для создания КА
{
    if (aWordSize)
        for (size_t i = 0; i != AlphaPower; ++i)//Генерирование всех половин палиндромов
        {
            *aFirstHalf = i;//Записываем первую половину в стек
            if (state_machine[i][enter].is_null())
                state_machine[i][enter] = state_generator::next();
            fill_state_machine<AlphaPower, nStates>(state_machine, aWordSize - 1, state_machine[i][enter], aFirstHalf + 1);
        }
    else
    {
        state cur_state = enter;
        while (*(--aFirstHalf) != STACK_EMPTY_VAL)//Раскручивание стека
        {
            if (state_machine[*aFirstHalf][cur_state].is_null())
                state_machine[*aFirstHalf][cur_state] = state_generator::next();
            if (*(aFirstHalf - 1) == STACK_EMPTY_VAL)//Если это был последний символ
                state_machine[*aFirstHalf][cur_state].set_available();//Установим конечное состояние для данной цепи
            cur_state = state_machine[*aFirstHalf][cur_state];
        }
    }
}

template<size_t AlphaPower, size_t nStates>
state_machine_t<AlphaPower, nStates>* create_state_machine(uint16_t aWordSize)//Функция для создания КА
{
    auto palindrom_state_machine = new state_machine_t<AlphaPower, nStates>;
    char stack[6] = { STACK_EMPTY_VAL, 0, 0, 0, 0, 0 };
    for (uint16_t CurWordSize = aWordSize / 2; CurWordSize; --CurWordSize)//Перебор всех палиндромов длины, не превыщающей заданной
        fill_state_machine<AlphaPower, nStates>(*palindrom_state_machine, CurWordSize, ENTER_STATE, stack + 1);
    for (size_t i = 0; i != AlphaPower; ++i)//Все слова длины 1 - палиндромы
        (*palindrom_state_machine)[i][ENTER_STATE].set_available();
    std::cout << state_generator::peek();
    state_generator::reset();
    return palindrom_state_machine;
}

char* read_from_file(const char* filename)//Чтение данных из файла
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

const char* next_lexem(const char* pos)//Перейти к следующей лексеме
{
    while (std::isspace(*pos))
        ++pos;
    return pos;
}

const char* skip_lexem(const char* pos)//Пропустить текущую лексему
{
    while (*pos && !std::isspace(*pos))
        ++pos;
    return pos;
}

bool islexend(const char* pos)//Проверка на достижение конца лексемы
{
    return std::isspace(*pos) || !(*pos);
}

bool is_ru(char letter)
{
    return false;//('А' <= letter && letter <= 'я') || letter == 'ё' || letter == 'Ё';
}

size_t ru_to_index(char letter)//Перевод буквы русского алфавита в индекс
{
    //Регистр не учитывается
    // if ('А' <= letter && letter <= 'Я')
    //     return letter + 'A' - 1;
    // if ('а' <= letter && letter <= 'я')
    //     return letter - 'а';
    // if (letter == 'Ё' || letter == 'ё')
    //     return 32;
    return 33;
}

std::vector<const char*> text_processing(const char* content)//Обработка текста
{
    std::unique_ptr<state_machine_t<RU_ALPHABET_POWER, RU_PSM_STATES_NUM>> psm_table { create_state_machine<RU_ALPHABET_POWER, RU_PSM_STATES_NUM>(6) };//создание автомата
    std::vector<const char*> result;
    const char* curpos = content;
    while (*curpos)
    {
        const char* lexbeg = curpos;//Начало текущей лексемы
        state curstate = ENTER_STATE;
        while (!curstate.is_null() && !islexend(curpos))
        {
            curstate = (*psm_table)[ru_to_index(*curpos)][curstate];
            ++curpos;
        }
        if (curstate.is_available())//Если достигли конечного состояния
            result.emplace_back(lexbeg);//То добавляем в список слов
        else if (curstate.is_null())//Если вышли из цикла раньше
            curpos = skip_lexem(curpos);//то пропускаем текущую лексему
        curpos = next_lexem(curpos);//Переход к следующей лексеме
    }
    return result;
}

void save(const std::vector<const char*>& lexems, const char* filename)//Функция для записи результатов в файл
{
    std::ofstream ofs { filename };
    for (auto i = lexems.cbegin(); i != lexems.cend(); ++i)
    {
        for (const char* cur = *i; !islexend(cur) && is_ru(*cur); ++cur)
            ofs << *cur;
        if (i != lexems.cend() - 1)
            ofs << ' ';
    }
    ofs.close();
}

int main()
{
    setlocale(LC_ALL, "RU");
    char* content = read_from_file("input.txt");
    auto result = text_processing(content);
    save(result, "output.txt");
    delete[] content;
}