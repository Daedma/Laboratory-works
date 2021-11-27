#pragma once
#include "directions.hpp"
#include <cstdint>
#include <utility>

class Player
{
public:
    using coord_t = std::pair<size_t, size_t>;
    Player(const coord_t& aCoord, int32_t aHp, int32_t aDmg) noexcept;
    const coord_t& move(directions aDir) noexcept;
    const coord_t& get_pos() const noexcept { return coord; }
    int32_t get_hp() const noexcept { return hp; }
    int32_t break_sword(int32_t aDmg) noexcept;
    int32_t take_dmg(int32_t aDmg) noexcept;
    int32_t deal_dmg() const noexcept;
    void print_status() const;
    bool changed() const noexcept { return changes; }

private:
    coord_t coord;
    int32_t hp;
    struct
    {
        int32_t cond, dmg;
    } sword;
    mutable bool changes = false;
};