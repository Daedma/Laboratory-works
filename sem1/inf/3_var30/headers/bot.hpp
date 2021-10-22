#pragma once
#include "objects.hpp"

class gameField;

class ttBot final
{
public:
    ttBot(gameField&, fieldObjects = fieldObjects::NOUGHT);
    void step();
    void reset(gameField&);
    void reset();
private:
    gameField& arena;
    //ttStrategy curStrat;

};