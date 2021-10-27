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
    bool clear(size_t, size_t) noexcept;
    bool valid() const { return _Valid; }
    char sym(size_t, size_t) const;
    void reset();
    fieldObjects check() const;
    fieldObjects at(size_t nRow, size_t nColumn) const noexcept { return _Arena[nRow][nColumn]; }
    bool setO(size_t nRow, size_t nColumn) noexcept { return setObj(fieldObjects::NOUGHT, nRow, nColumn); }
    bool setX(size_t nRow, size_t nColumn) noexcept { return setObj(fieldObjects::CROSS, nRow, nColumn); }
    bool setObj(fieldObjects, size_t, size_t) noexcept;
};

std::ostream& operator<<(std::ostream&, const gameField&);