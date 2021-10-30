#include "../headers/field.hpp"
#include <iostream>
#define FILLING_CUP 9

#if (defined(_WIN32) || defined(_WIN64))
#define AGGREGATE char(176)
#endif
#if (defined(LINUX) || defined(__linux__))
#define AGGREGATE ' '
#endif

gameField::gameField() : _Valid{true}, _Filling{0}
{
    for (auto &y : _Arena)
        for (auto &x : y)
            x = fieldObjects::EMPTY;
}

bool gameField::setObj(fieldObjects Obj, size_t nRow, size_t nColumn) noexcept
{
    if (nRow > 2 || nColumn > 2 || _Arena[nRow][nColumn] != fieldObjects::EMPTY)
        return false;
    _Arena[nRow][nColumn] = Obj;
    ++_Filling;
    if (_Filling == FILLING_CUP)
        _Valid = false;
    return true;
}

bool gameField::clear(size_t nRow, size_t nColumn) noexcept
{
    if (nRow > 2 || nColumn > 2 || _Arena[nRow][nColumn] == fieldObjects::EMPTY)
        return false;
    _Arena[nRow][nColumn] = fieldObjects::EMPTY;
    --_Filling;
    _Valid = true;
    return true;
}

void gameField::reset()
{
    for (auto &y : _Arena)
        for (auto &x : y)
            x = fieldObjects::EMPTY;
    _Filling = 0;
    _Valid = true;
}

fieldObjects gameField::check() const
{
    struct counter
    {
        size_t nought, cross;

        size_t add(fieldObjects _Obj)
        {
            if (_Obj == fieldObjects::NOUGHT)
            {
                ++nought;
                return nought;
            }
            if (_Obj == fieldObjects::CROSS)
            {
                ++cross;
                return cross;
            }
            return 0;
        }

        fieldObjects goal() const
        {
            if (nought == 3)
                return fieldObjects::NOUGHT;
            if (cross == 3)
                return fieldObjects::CROSS;
            return fieldObjects::EMPTY;
        }

        void reset()
        {
            nought = 0;
            cross = 0;
        }
    };

    counter _Count;
    //перебор всех вариантов
    //1nd case
    for (auto &y : _Arena)
    {
        for (auto &x : y)
        {
            _Count.add(x);
        }
        if (_Count.goal() != fieldObjects::EMPTY)
            return _Count.goal();
        else
            _Count.reset();
    }

    //2nd case
    for (auto x = 0; x != 3; ++x)
    {
        for (auto y = 0; y != 3; ++y)
        {
            _Count.add(_Arena[y][x]);
        }
        if (_Count.goal() != fieldObjects::EMPTY)
            return _Count.goal();
        else
            _Count.reset();
    }

    //3nd case
    for (auto d = 0; d != 3; ++d)
    {
        _Count.add(_Arena[d][d]);
    }
    if (_Count.goal() != fieldObjects::EMPTY)
        return _Count.goal();
    else
        _Count.reset();

    //4nd case
    for (auto dy = 2, dx = 0; dx != 3; --dy, ++dx)
    {
        _Count.add(_Arena[dy][dx]);
    }
    if (_Count.goal() != fieldObjects::EMPTY)
        return _Count.goal();

    return fieldObjects::EMPTY;
}

char gameField::sym(size_t nRow, size_t nColumn) const
{
    if (_Arena[nRow][nColumn] == fieldObjects::CROSS)
        return 'X';
    if (_Arena[nRow][nColumn] == fieldObjects::NOUGHT)
        return 'O';
    return AGGREGATE;
}

std::ostream &operator<<(std::ostream &os, const gameField &_Field)
{
    os << '|' << '\\' << "|1|2|3|" << std::endl;
    os << "|1|" << _Field.sym(0, 0) << '|' << _Field.sym(0, 1) << '|' << _Field.sym(0, 2) << '|' << std::endl;
    os << "|2|" << _Field.sym(1, 0) << '|' << _Field.sym(1, 1) << '|' << _Field.sym(1, 2) << '|' << std::endl;
    os << "|3|" << _Field.sym(2, 0) << '|' << _Field.sym(2, 1) << '|' << _Field.sym(2, 2) << '|' << std::endl;
    return os;
}