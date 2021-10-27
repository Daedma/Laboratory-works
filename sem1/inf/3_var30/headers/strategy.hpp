#pragma once
#include "..\headers\objects.hpp"
#include <array>
#include <stack>
#include <utility>

class gameField;

//класс, представляющий стратегию, по которой играет бот
class gameStrategy
{
    using steps_type = std::stack<std::pair<uint16_t, uint16_t>>;
public:
    gameStrategy(const gameField&, fieldObjects);
    //возвращает следующий шаг, который надо сделать
    steps_type::value_type step() noexcept;
    //вернуть количество шагов до победы
    uint16_t left() const noexcept;
    //обновить стратегии в соответствии с изменением поля
    //необходимо вызывать после каждого хода противника
    void update();
    //сбросить стратегию
    void reset(fieldObjects);
private:
    //значение номера стратегии, при которой ходы делаются случайным образом
    static constexpr size_t randval = static_cast<size_t>(-1);
    //шаги, необходимые для победы
    static const std::array<std::array<steps_type::value_type, 3>, 8> Intents;

    const gameField& Arena;
    fieldObjects Side;
    size_t CurStrat;//номер текущей стратегии
    steps_type StepsQueue;//очередь шагов

    //заполнить очередь 
    void fill();
    //заполнить очередь рандомными шагами
    void randfill();
    //проверить валидность текущей стратегии
    bool fit() const noexcept;
    //узнать приоритет для данной стратегии
    std::pair<bool, uint16_t> priority(size_t) const noexcept;
    //обновить стратегию в соответствии обстановкой на поле
    void change();
    //узнать, является ли текущая стратегия рандомной
    bool random() const noexcept { return CurStrat == randval; }
    //очистить очередь
    void clear() noexcept { while (!StepsQueue.empty()) StepsQueue.pop(); }
    //выбрать стратегию
    size_t select() const;
};