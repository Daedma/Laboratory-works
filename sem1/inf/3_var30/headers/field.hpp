#pragma once
#include <array>
#include <iosfwd>
#include "objects.hpp"


//0 1 2
//1 * *
//2 * *
class gameField
{
    using gameFieldType = std::array<std::array<fieldObjects, 3>, 3>;
    gameFieldType _Arena;
    bool _Valid;
    size_t _Filling;
public:
    gameField();
    bool setO(size_t, size_t);
    bool setX(size_t, size_t);
    bool clear(size_t, size_t);
    bool valid() const { return _Valid; }
    char sym(size_t, size_t) const;
    fieldObjects at(size_t, size_t) const;
    void reset();
    fieldObjects check() const;
};

std::ostream& operator<<(std::ostream&, const gameField&);