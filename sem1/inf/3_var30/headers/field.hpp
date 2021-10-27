#pragma once
#include <array>
#include <iosfwd>
#include "objects.hpp"


//Класс, представляющий собой игровое поле
class gameField
{
    using gameFieldType = std::array<std::array<fieldObjects, 3>, 3>;
    gameFieldType _Arena;
    bool _Valid;//есть ли свободные места на поле
    size_t _Filling;//заполненность поля
public:
    gameField();
    //очистить координату
    bool clear(size_t, size_t) noexcept;
    //есть ли на поле ещё свободные места
    bool valid() const { return _Valid; }
    //вернуть символ объекта по этой координате
    char sym(size_t, size_t) const;
    //сбросить поле к изначальному состоянию
    void reset();
    //проверить, кто победил
    fieldObjects check() const;
    //доступ к элементу по координате (номер строки, номер столбца)
    fieldObjects at(size_t nRow, size_t nColumn) const noexcept { return _Arena[nRow][nColumn]; }
    //установить нолик по заданной координате
    bool setO(size_t nRow, size_t nColumn) noexcept { return setObj(fieldObjects::NOUGHT, nRow, nColumn); }
    //учстановить крестик по заданной координате
    bool setX(size_t nRow, size_t nColumn) noexcept { return setObj(fieldObjects::CROSS, nRow, nColumn); }
    //установить переданный объект по заданной координате
    bool setObj(fieldObjects, size_t, size_t) noexcept;
};

//выводит поле
std::ostream& operator<<(std::ostream&, const gameField&);