#pragma once
#include "objects.hpp"
#include "strategy.hpp"

class gameField;

class gameBot final
{
public:
    gameBot(gameField&, fieldObjects = fieldObjects::NOUGHT);
    void step();
    void reset(fieldObjects);
    void reset();
private:
    using coord_type = decltype(std::declval<gameStrategy>().step());
    gameField& Arena;
    fieldObjects Side;
    gameStrategy curStrat;

    std::pair<bool, coord_type> danger() const noexcept;
};