#pragma once
#include "..\headers\objects.hpp"
#include <array>
#include <stack>
#include <utility>

class gameField;

class gameStrategy
{
    using steps_type = std::stack<std::pair<uint16_t, uint16_t>>;
public:
    gameStrategy(const gameField&, fieldObjects);
    steps_type::value_type step() noexcept;
    uint16_t left() const noexcept;
    void update();
    void reset(fieldObjects);
private:
    static constexpr size_t randval = static_cast<size_t>(-1);

    static const std::array<std::array<steps_type::value_type, 3>, 8> Intents;

    const gameField& Arena;
    fieldObjects Side;
    size_t CurStrat;
    steps_type StepsQueue;

    void fill();
    void randfill();
    bool fit() const noexcept;
    std::pair<bool, uint16_t> priority(size_t) const noexcept;
    void change();
    inline bool random() const noexcept;
    inline void clear() noexcept;
    size_t select() const;
};