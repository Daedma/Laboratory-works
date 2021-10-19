#include "..\headers\field.hpp"
#include <iostream>
#define FILLING_CUP 9
#define AGGREGATE char(176)

ttField::ttField() :
    _Valid { true }, _Filling { 0 }
{
    for (auto& y : _Arena)
        for (auto& x : y)
            x = ttObjType::EMPTY;
}

bool ttField::setO(size_t _X, size_t _Y)
{
    if (_X > 2 || _Y > 2 || _Arena[_Y][_X] != ttObjType::EMPTY)
        return false;
    _Arena[_Y][_X] = ttObjType::NOUGHT;
    ++_Filling;
    if (_Filling == FILLING_CUP)
        _Valid = false;
    return true;
}

bool ttField::setX(size_t _X, size_t _Y)
{
    if (_X > 2 || _Y > 2 || _Arena[_Y][_X] != ttObjType::EMPTY)
        return false;
    _Arena[_Y][_X] = ttObjType::CROSS;
    ++_Filling;
    if (_Filling == FILLING_CUP)
        _Valid = false;
    return true;
}

bool ttField::clear(size_t _X, size_t _Y)
{
    if (_X > 2 || _Y > 2 || _Arena[_Y][_X] == ttObjType::EMPTY)
        return false;
    _Arena[_Y][_X] = ttObjType::EMPTY;
    --_Filling;
    if (_Filling != FILLING_CUP)
        _Valid = true;
    return true;
}

void ttField::reset()
{
    for (auto& y : _Arena)
        for (auto& x : y)
            x = ttObjType::EMPTY;
    _Filling = 0;
    _Valid = true;
}

ttObjType ttField::check() const
{
    struct counter
    {
        size_t nought, cross;

        size_t add(ttObjType _Obj)
        {
            if (_Obj == ttObjType::NOUGHT)
            {
                ++nought;
                return nought;
            }
            if (_Obj == ttObjType::CROSS)
            {
                ++cross;
                return cross;
            }
            return 0;
        }

        ttObjType goal() const
        {
            if (nought == 3)
                return ttObjType::NOUGHT;
            if (cross == 3)
                return ttObjType::CROSS;
            return ttObjType::EMPTY;
        }

        void reset()
        {
            nought = 0;
            cross = 0;
        }
    };

    counter _Count;

    //1nd case
    for (auto& y : _Arena)
    {
        for (auto& x : y)
        {
            _Count.add(x);
        }
        if (_Count.goal() != ttObjType::EMPTY)
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
        if (_Count.goal() != ttObjType::EMPTY)
            return _Count.goal();
        else
            _Count.reset();
    }

    //3nd case
    for (auto d = 0; d != 3; ++d)
    {
        _Count.add(_Arena[d][d]);
    }
    if (_Count.goal() != ttObjType::EMPTY)
        return _Count.goal();
    else
        _Count.reset();

    //4nd case
    for (auto dy = 2, dx = 0; dx != 3; --dy, ++dx)
    {
        _Count.add(_Arena[dy][dx]);
    }
    if (_Count.goal() != ttObjType::EMPTY)
        return _Count.goal();

    return ttObjType::EMPTY;
}

char ttField::sym(size_t _X, size_t _Y) const
{
    if (_Arena[_Y][_X] == ttObjType::CROSS)
        return 'X';
    if (_Arena[_Y][_X] == ttObjType::NOUGHT)
        return 'O';
    return AGGREGATE;
}
//| |0|1|2|
//|0| | | |
//|1| | | |
//|2| | | |
std::ostream& operator<<(std::ostream& os, const ttField& _Field)
{
    os << '|' << char(219) << "|0|1|2|" << std::endl;
    os << "|0|" << _Field.sym(0, 0) << '|' << _Field.sym(1, 0) << '|' << _Field.sym(2, 0) << '|' << std::endl;
    os << "|1|" << _Field.sym(0, 1) << '|' << _Field.sym(1, 1) << '|' << _Field.sym(2, 1) << '|' << std::endl;
    os << "|2|" << _Field.sym(0, 2) << '|' << _Field.sym(1, 2) << '|' << _Field.sym(2, 2) << '|' << std::endl;
    return os;
}