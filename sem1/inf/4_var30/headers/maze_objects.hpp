#pragma once
#include <cstdint>
#include <string_view>
#include "directions.hpp"

class Player;

class MazeObject
{
public:
    MazeObject(int32_t aHp, int32_t aDmg) noexcept;
    virtual ~MazeObject() = default;
    MazeObject(MazeObject&&) = default;
    MazeObject(const MazeObject&) = default;
    MazeObject& operator=(const MazeObject&) = default;
    MazeObject& operator=(MazeObject&&) = default;
    virtual bool need_clear() const noexcept = 0 { return !hp; }
    virtual std::string_view take_dmg(Player& aPlayer) noexcept = 0;
    virtual std::string_view try_move(Player& aPlayer, directions aDir) noexcept = 0;
    virtual char sym() const noexcept = 0;
protected:
    int32_t hp;
    int32_t dmg;
};

class Pass : public MazeObject
{
public:
    Pass() noexcept;
    bool need_clear() const noexcept override { return false; }
    std::string_view take_dmg(Player& aPlayer) noexcept override;
    std::string_view try_move(Player& aPlayer, directions aDir) noexcept override;
    char sym() const noexcept override { return symbol; }

    static constexpr char symbol = '.';
};

class Wall : public MazeObject
{
public:
    Wall(int32_t aHp, int32_t aDmg) noexcept;
    std::string_view take_dmg(Player& aPlayer) noexcept override;
    std::string_view try_move(Player& aPlayer, directions aDir) noexcept override;
    char sym() const noexcept override { return symbol; }

    static constexpr char symbol = '#';
};

class Monster : public MazeObject
{
public:
    Monster(int32_t aHp, int32_t aDmg) noexcept;
    std::string_view take_dmg(Player& aPlayer) noexcept override;
    std::string_view try_move(Player& aPlayer, directions aDir) noexcept override;
    char sym() const noexcept override { return symbol; }

    static constexpr char symbol = 'M';
};