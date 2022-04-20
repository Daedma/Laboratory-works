#include <fstream>
#include <cctype>
#include <array>
#include <vector>
#include <iterator>
#include <algorithm>

//Состояния
enum States
{
    NULL_STATE, ENTER_STATE,
    SIGN_STATE, DIGIT_STATE,
    ALPHA_STATE, ALNUM_STATE1, ALNUM_STATE2, ALNUM_STATE3, ALNUM_STATE4,
    SEP_STATE,
    OPERATOR_LESS_STATE, OPERATOR_GREAT_STATE, OPERATOR_EQUAL_STATE, OPERATOR_EQUAL_STATE2, OPERATOR_GREAT_STATE2
};
//Символы алфавита
enum Symbols { ALPHA, DIGIT, SPACE, SIGN, OPERATOR_LESS, OPERATOR_GREAT, OPERATOR_EQUAL, UNDEFINED };
//Типы лексем
enum Lexem_type { KW, CO, EQ, AO, WL, VL, ID };

Symbols get_sym(char c)//Определить тип символа
{
    if (std::isspace(c) || c == 0) return SPACE;
    if (std::isalpha(c)) return ALPHA;
    if (std::isdigit(c)) return DIGIT;
    if (c == '-' || c == '+') return SIGN;
    if (c == '<') return OPERATOR_LESS;
    if (c == '>') return OPERATOR_GREAT;
    if (c == '=') return OPERATOR_EQUAL;
    return UNDEFINED;
}

struct lexem//Класс лексемы
{
    const char* begin;//Начало лексемы
    size_t length;//Длина лексемы
    Lexem_type id;//Тип лексемы
};

void skipws(const char*& pos)//Пропустить пробелы
{
    while (std::isspace(*pos) && *pos) ++pos;
}

int pow(int base, int exp)//целочисленное возведение в степень
{
    if (exp == 1) return base;
    if (exp % 2) return pow(base, exp - 1) * base;
    int half = pow(base, exp / 2);
    return half * half;
}

bool valid_num(const char* numstr, size_t sz)//Проверка числа на допустимые значения
{
    if (sz > 5) return false;
    size_t num = 0;
    while (sz)
    {
        num += pow(10, sz) * (*numstr - '0');
        --sz;
        ++numstr;
    }
    return num <= 32768;
}

bool strcomp(const char* lhs, const char* rhs, size_t sz)//Проверка на точное равенство найденной лексемы и строки
{
    while (sz && *lhs)
    {
        if (*lhs++ != *rhs++)
            return false;
        --sz;
    }
    return !sz;
}

Lexem_type lextype(const char* word, size_t word_size, States state)//Определение типа лексемы
{
    static constexpr std::array<const char*, 4> key_words = { "if", "then", "else", "end" };
    switch (state)
    {
    case NULL_STATE:
        return WL;
        break;
    case SIGN_STATE:
        return AO;
    case DIGIT_STATE:
        if (valid_num(word, word_size))
            return VL;
        else
            return WL;
    case ALPHA_STATE:
        return ID;
    case ALNUM_STATE1:
        if (strcomp(key_words[0], word, word_size))
            return KW;
        return ID;
    case ALNUM_STATE2:
        if (strcomp(key_words[3], word, word_size))
            return KW;
        return ID;
    case ALNUM_STATE3:
        if (strcomp(key_words[1], word, word_size) || strcomp(key_words[2], word, word_size))
            return KW;
        return ID;
    case ALNUM_STATE4:
        return ID;
    case OPERATOR_GREAT_STATE2:
    case OPERATOR_EQUAL_STATE2:
    case OPERATOR_GREAT_STATE:
    case OPERATOR_LESS_STATE:
        return CO;
    case OPERATOR_EQUAL_STATE:
        return EQ;
    default:
        break;
    }
    return WL;
}

void lexical_analysis(const char* text, std::vector<lexem>& results)
{

    using state_machine_t = std::array<std::array<States, 8>, 15>;//[STATE][SYMBOL]

    static constexpr const state_machine_t  state_machine = //Таблица переходов
    { {
            // ALPHA, DIGIT, SPACE, SIGN, OPERATOR_LESS, OPERATOR_GREAT, OPERATOR_EQUAL, UNDEFINED 
        { NULL_STATE, NULL_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//NULL_STATE
        { ALPHA_STATE, DIGIT_STATE, SEP_STATE, SIGN_STATE, OPERATOR_LESS_STATE, OPERATOR_GREAT_STATE, OPERATOR_EQUAL_STATE, NULL_STATE },//ENTER_STATE
        { SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//SIGN_STATE
        { NULL_STATE, DIGIT_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//DIGIT_STATE
        { ALNUM_STATE1, ALNUM_STATE1, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//ALPHA_STATE
        { ALNUM_STATE2, ALNUM_STATE2, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//ALNUM_STATE1
        { ALNUM_STATE3, ALNUM_STATE3, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//ALNUM_STATE2
        { ALNUM_STATE4, ALNUM_STATE4, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//ALNUM_STATE3
        { NULL_STATE, NULL_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//ALNUM_STATE4
        { ALPHA_STATE, DIGIT_STATE, SEP_STATE, SIGN_STATE, OPERATOR_LESS_STATE, OPERATOR_GREAT_STATE, OPERATOR_EQUAL_STATE, NULL_STATE },//SEP_STATE
        { SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, OPERATOR_GREAT_STATE2, OPERATOR_EQUAL_STATE2, NULL_STATE },//OPERATOR_LESS_STATE
        { SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, OPERATOR_EQUAL_STATE2, NULL_STATE },//OPERATOR_GREAT_STATE
        { SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//OPERATOR_EQUAL_STATE
        { SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE },//OPERATOR_EQUAL_STATE2
        { SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, SEP_STATE, NULL_STATE } //OPERATOR_GREAT_STATE2
        } };

    const char* curpos = text;//Текущая позиция
    skipws(curpos);
    while (*curpos)//Пока не достигли конца строки
    {
        const char* lexbeg = curpos;//Начало считываемой лексемы
        size_t cursize = 0;//Размер текущей лексемы
        States curstate = ENTER_STATE,//Текущее состояние
            prevstate;//Предыдущее состояние
        while (curstate != SEP_STATE)//Переход по состояниям пока не встретится разделитель
        {
            prevstate = curstate;
            curstate = state_machine[curstate][get_sym(*curpos++)];
            ++cursize;
        }
        --curpos;
        --cursize;
        results.push_back({ lexbeg, cursize, lextype(lexbeg, cursize, prevstate) });//Добавим лексему в список
        skipws(curpos);
    }
}

const char* lexid_c(Lexem_type id)//Текстовое представление типа лексемы
{
    switch (id)
    {
    case KW:
        return "[kw]";
    case CO:
        return "[co]";
    case EQ:
        return "[eq]";
    case AO:
        return "[ao]";
    case WL:
        return "[wl]";
    case VL:
        return "[vl]";
    case ID:
        return "[id]";
    }
    return "";
}

std::ostream& operator<<(std::ostream& os, const lexem& rhs)//Вывод лексемы
{
    return os.write(rhs.begin, rhs.length) << lexid_c(rhs.id);
}

const char* read(const char* filename)//Считать содержимое файла в строку
{
    std::ifstream ifs { filename, std::ios::binary };
    ifs.seekg(0, std::ios::end);
    size_t filesize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    char* content = new char[filesize + 1];
    ifs.getline(content, filesize + 1, '\0');
    ifs.close();
    return content;
}

void save(const char* filename, const std::vector<lexem>& lexems)//Записать результат работы программы в файл
{
    std::ofstream ofs { filename };
    std::copy(lexems.cbegin(), lexems.cend(), std::ostream_iterator<lexem>{ofs, " "});
    ofs << '\n';
    for (const auto& i : lexems)
    {
        if (i.id == ID)
            ofs.write(i.begin, i.length) << ' ';
    }
    ofs << '\n';
    for (const auto& i : lexems)
    {
        if (i.id == VL)
            ofs.write(i.begin, i.length) << ' ';
    }
    ofs.close();
}

int main()
{
    setlocale(LC_ALL, "RU");
    std::vector<lexem> lexems;
    const char* content = read("input.txt");
    lexical_analysis(content, lexems);
    save("output.txt", lexems);
    delete[] content;
}