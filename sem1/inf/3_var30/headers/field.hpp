#pragma once
#include <array>
#include <iosfwd>
#include "objects.hpp"


//0 1 2
//1 * *
//2 * *
class ttField
{
    using ttFieldType = std::array<std::array<ttObjType, 3>, 3>;
    ttFieldType _Arena;
    bool _Valid;
    size_t _Filling;
public:
    ttField();
    bool setO(size_t, size_t);
    bool setX(size_t, size_t);
    bool clear(size_t, size_t);
    bool valid() const { return _Valid; }
    char sym(size_t, size_t) const;
    void reset();
    ttObjType check() const;
};

std::ostream& operator<<(std::ostream&, const ttField&);