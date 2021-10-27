#pragma once
#include "objects.hpp"
#include "strategy.hpp"

class gameField;

//класс, представляющий бота
class gameBot final
{
public:
    gameBot(gameField&, fieldObjects = fieldObjects::NOUGHT);
    //сделать ход ботом
    void step();
    //узнать, за какую сторону играет бот
    fieldObjects get_side() const noexcept { return Side; }
    //сбросить бота к начальному состоянию
    void reset(fieldObjects);
    void reset();
private:
    using coord_type = decltype(std::declval<gameStrategy>().step());
    gameField& Arena;
    fieldObjects Side;//сторона
    gameStrategy curStrat;//текущая стратегия

    //проверить поле на наличие предвыигрышной ситуации у противника
    std::pair<bool, coord_type> is_danger() const noexcept;
};