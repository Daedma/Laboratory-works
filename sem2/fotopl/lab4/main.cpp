#include <fstream>
#include <cctype>
#include <array>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cstring>

/*
if <логическое выражение> then
<операторы>
[ else <операторы> ]
end

Уровень 1
<логическое выражение> → <операнд> | <операнд><операция сравнения><операнд>
<операция сравнения> →| <= | >= | = | <>
<операнд> → <идентификатор>|<константа>
<операторы> → <идентификатор> = <арифметическое выражение>
<арифметическое выражние> → <операнд> |
<арифметическое выражение><арифметическая операция><операнд>
<арифметическая операция> → + | –
*/

class Lexical_analyser
{
    enum Symbols
    {
        ALPHA,
        DIGIT,
        SPACE,
        ARITH,
        MINUS,
        LESS,
        GREAT,
        EQUAL,
        UNDEFINED,
        SYMCOUNT
    };

    enum States
    {
        NULL_STATE,
        ENTER_STATE,
        ARITH_STATE,
        DIGIT_STATE,
        ALNUM_STATE,
        LESS_STATE,
        GREAT_STATE,
        EQUAL_STATE,
        GREAT_STATE2,
        MINUS_STATE,
        STATECOUNT
    };

public:
    struct Token //Класс лексемы
    {
        enum Types
        {
            IF,
            TH,
            EL,
            EN,
            CO,
            EQ,
            ID,
            VL,
            AO,
            WL,
            TYPECOUNT
        };
        const char *begin; //Начало лексемы
        size_t length;     //Длина лексемы
        Types id;          //Тип лексемы
    };

    static const char* to_str(Token::Types type)
    {
        switch(type)
        {
            case Token::IF:
                return "if";
            case Token::TH:
                return "th";
            case Token::EL:
                return "el";
            case Token::EN:
                return "en";
            case Token::CO:
                return "co";
            case Token::EQ:
                return "eq";
            case Token::ID:
                return "id";
            case Token::VL:
                return "vl";
            case Token::AO:
                return "ao";
            case Token::WL:
                return "wl";
        }
        return "";
    }

    static std::vector<Token> get_tokens(const char *text)
    {
        std::vector<Token> tokens;
        const char *curpos = text; //Текущая позиция
        skipws(curpos);
        while (*curpos) //Пока не достигли конца строки
        {
            const char *lexbeg = curpos;   //Начало считываемой лексемы
            size_t cursize = 0;            //Размер текущей лексемы
            States curstate = ENTER_STATE, //Текущее состояние
                prevstate;                 //Предыдущее состояние
            do
            {
                prevstate = curstate;
                curstate = transition_table[charid(*curpos)][curstate];
                ++cursize;
            } while (curstate != ENTER_STATE); //Переход по состояниям пока не встретится разделитель
            --curpos;
            --cursize;
            tokens.emplace_back(get_token(lexbeg, cursize, prevstate)); //Добавим лексему в список
            skipws(curpos);
        }
        return tokens;
    };

    Lexical_analyser() = delete;

private:
    using trans_table_t = std::array<std::array<States, STATECOUNT>, SYMCOUNT>; //[SYMBOL][STATE]

    static constexpr const trans_table_t transition_table = //Таблица переходов
        {{
          // NULL_STATE   ENTER_STATE  ARITH_STATE  DIGIT_STATE  ALNUM_STATE  LESS_STATE    GREAT_STATE  EQUAL_STATE  GREAT_STATE2 MINUS_STATE
            {NULL_STATE,  ALNUM_STATE, ENTER_STATE, NULL_STATE,  ALNUM_STATE, ENTER_STATE,  ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE},  // ALPHA
            {NULL_STATE,  DIGIT_STATE, ENTER_STATE, DIGIT_STATE, ALNUM_STATE, ENTER_STATE,  ENTER_STATE, ENTER_STATE, ENTER_STATE, DIGIT_STATE},  // DIGIT
            {ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE,  ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE},  // SPACE
            {ENTER_STATE, ARITH_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE,  ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE},  // ARITH
            {ENTER_STATE, MINUS_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE,  ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE},  // MINUS
            {ENTER_STATE, LESS_STATE,  ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE,  ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE},  // LESS
            {ENTER_STATE, GREAT_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE, GREAT_STATE2, ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE},  // GREAT
            {ENTER_STATE, EQUAL_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE, EQUAL_STATE,  EQUAL_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE},  // EQUAL
            {NULL_STATE,  NULL_STATE,  ENTER_STATE, NULL_STATE,  NULL_STATE,  ENTER_STATE,  ENTER_STATE, ENTER_STATE, ENTER_STATE, ENTER_STATE},  // UNDEFINED
        }};

    static Symbols charid(char c)
    {
        if (std::isspace(c) || c == 0)
            return SPACE;
        if (std::isalpha(c))
            return ALPHA;
        if (std::isdigit(c))
            return DIGIT;
        if (c == '+')
            return ARITH;
        if (c == '-')
            return MINUS;
        if (c == '<')
            return LESS;
        if (c == '>')
            return GREAT;
        if (c == '=')
            return EQUAL;
        return UNDEFINED;
    }

    static int pow(int base, int exp) //целочисленное возведение в степень
    {
        if (exp == 1)
            return base;
        if (exp % 2)
            return pow(base, exp - 1) * base;
        int half = pow(base, exp / 2);
        return half * half;
    }

    static bool isval(const char *numstr, size_t sz) //Проверка числа на допустимые значения
    {
        if (sz > 5)
            return false;
        bool negate = false;
        if (numstr[0] == '-')
        {
            negate = true;
            ++numstr;
            --sz;
        }
        int num = 0;
        while (sz)
        {
            num += pow(10, sz) * (*numstr - '0');
            --sz;
            ++numstr;
        }
        return num <= 32767 + negate;
    }

    static bool strcmp(const char *lhs, const Token &rhs) //Проверка на точное равенство найденной лексемы и строки
    {
        size_t left = rhs.length;
        const char *it = rhs.begin;
        while (left && *lhs)
        {
            if (*lhs++ != *it++)
                return false;
            --left;
        }
        return !left;
    }

    static void skipws(const char *&pos)
    {
        while (std::isspace(*pos) && *pos)
            ++pos;
    }

    static Token get_token(const char *beg, size_t sz, States st)
    {
        switch (st)
        {
        case NULL_STATE:
            return {beg, sz, Token::WL};
        case MINUS_STATE:
        case ARITH_STATE:
            return {beg, sz, Token::AO};
        case DIGIT_STATE:
            if (isval(beg, sz))
                return {beg, sz, Token::VL};
            else
                return {beg, sz, Token::WL};
        case ALNUM_STATE:
        {
            Token tmp{beg, sz, Token::ID};
            if (sz > 5)
                tmp.id = Token::WL;
            else if (strcmp("if", tmp))
                tmp.id = Token::IF;
            else if (strcmp("then", tmp))
                tmp.id = Token::TH;
            else if (strcmp("else", tmp))
                tmp.id = Token::EL;
            else if (strcmp("end", tmp))
                tmp.id = Token::EN;
            return tmp;
        }
        case LESS_STATE:
        case GREAT_STATE:
        case EQUAL_STATE:
        case GREAT_STATE2:
            return {beg, sz, Token::CO};
        }
        return {beg, sz, Token::WL};
    }
};

class Syntax_analyzer
{
    enum States
    {
        ERROR_STATE,
        ENTER,
        IF,
        LOGEXP0,
        LOGEXP1,
        LOGEXP2,
        OPER0,
        OPER1,
        OPER2,
        OPER3,
        ELSE0,
        ELSE1,
        ELSE2,
        ELSE3,
        STATECOUNT
    };

    using trans_table_t = std::array<std::array<States, STATECOUNT>, Lexical_analyser::Token::TYPECOUNT>;
    static constexpr trans_table_t transition_table = {{
        //ERROR_STATE  ENTER        IF           LOGEXP0      LOGEXP1      LOGEXP2      OPER0        OPER1        OPER2        OPER3        ELSE0        ELSE1        ELSE2        ELSE3
        { ERROR_STATE, IF,          ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE },//IF
        { ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, OPER0,       ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE },//TH
        { ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ELSE0,       ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE },//EL
        { ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ENTER,       ERROR_STATE, ERROR_STATE, ERROR_STATE, ENTER       },//EN
        { ERROR_STATE, ERROR_STATE, ERROR_STATE, LOGEXP1,     ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE },//CO
        { ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, OPER2,       ERROR_STATE, ERROR_STATE, ERROR_STATE, ELSE2,       ERROR_STATE, ERROR_STATE },//EQ
        { ERROR_STATE, ERROR_STATE, LOGEXP0,     ERROR_STATE, LOGEXP2,     ERROR_STATE, OPER1,       ERROR_STATE, OPER3,       ERROR_STATE, ELSE1,       ERROR_STATE, ELSE3,       ERROR_STATE },//ID
        { ERROR_STATE, ERROR_STATE, LOGEXP0,     ERROR_STATE, LOGEXP2,     ERROR_STATE, ERROR_STATE, ERROR_STATE, OPER3,       ERROR_STATE, ERROR_STATE, ERROR_STATE, ELSE3,       ERROR_STATE },//VL
        { ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, OPER2,       ERROR_STATE, ERROR_STATE, ERROR_STATE, ELSE2       },//AO
        { ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE, ERROR_STATE },//WL
    }};

public:

    struct Report
    {
    private:
        char* _message;
        int64_t _errpos;

        static constexpr int NOERROR_POS = -1;
        static constexpr int ID_SIZE = 3;
        static constexpr int DEFAULT_MESSAGE_SIZE = (ID_SIZE + 1) * Lexical_analyser::Token::TYPECOUNT;

    public:

        Report(const char* mess) : 
            _message(new char[std::strlen(mess)]), _errpos(NOERROR_POS)
        {
            std::strcpy(_message, mess);
        }

        Report(int64_t pos, States last_state) : _errpos(pos)
        {
            _message = new char[DEFAULT_MESSAGE_SIZE];
            char* cursor = _message;
            for(size_t i = 0; i != transition_table.size(); ++i)
                if(transition_table[i][last_state] != ERROR_STATE)
                {
                    std::strcpy(cursor, Lexical_analyser::to_str(static_cast<Lexical_analyser::Token::Types>(i)));
                    cursor += ID_SIZE;
                }
        }

        ~Report() { delete[] _message; }

        void print(std::ostream& os) const
        {
            if(_errpos != NOERROR_POS) os <<  _errpos;
            os << _message;
        }
    };

    static Report get_report(const std::vector<Lexical_analyser::Token>& tokens)
    {
        States curstate = ENTER, prevstate;
        auto i = tokens.cbegin();
        while (i != tokens.cend() && curstate != ERROR_STATE)
        {
            prevstate = curstate;
            curstate = transition_table[i->id][curstate];
            ++i;
        }
        if (curstate == ENTER)
            return {"OK"};
        return {std::distance(tokens.cbegin(), i), prevstate};
    }

};

const char *read(const char *filename) //Считать содержимое файла в строку
{
    std::ifstream ifs{filename, std::ios::binary};
    ifs.seekg(0, std::ios::end);
    size_t filesize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    char *content = new char[filesize + 1];
    ifs.getline(content, filesize + 1, '\0');
    ifs.close();
    return content;
}

void save(const char *filename, const std::vector<Lexical_analyser::Token> &tokens, const Syntax_analyzer::Report& report) //Записать результат работы программы в файл
{
    std::ofstream ofs{filename};
    for(auto& i : tokens)
        ofs.write(i.begin, i.length) << "[" << Lexical_analyser::to_str(i.id) << "] ";
    ofs << '\n';
    report.print(ofs);
}

int main()
{
    setlocale(LC_ALL, "RU");
    const char* content = read("input.txt");
    auto tokens = Lexical_analyser::get_tokens(content);
    auto report = Syntax_analyzer::get_report(tokens);
    save("output.txt", tokens, report);
    delete[] content;
}