#pragma once
#include "..\headers\objects.hpp"
#include <memory>
#include <array>
#include <stack>
#include <utility>
#include <vector>

class gameField;

class gameStrategy
{
public:
    std::pair<uint16_t, uint16_t> step() const noexcept;
    inline uint16_t left() const noexcept;
private:
    using steps_type = std::stack<std::pair<uint16_t, uint16_t>>;

    class strategy_type
    {
    public:
        virtual void fill(steps_type&) const = 0;
        virtual bool fit(const gameField&) const = 0;
    protected:
        strategy_type(fieldObjects);
        static void shuffle(std::initializer_list<steps_type::value_type>, steps_type&);
        std::vector<steps_type::value_type> enemy_pos(const gameField&) const;
    private:
        fieldObjects Side;
    };

    class row : public strategy_type
    {
    public:
        static std::pair<uint16_t, std::unique_ptr<strategy_type>> suit(const gameField&, fieldObjects);
        row(uint16_t, fieldObjects);
        row(fieldObjects);
        void fill(steps_type&) const override;
        bool fit(const gameField&) const override;
    private:
        uint16_t nRow;
    };

    class column : public strategy_type
    {
    public:
        static std::pair<uint16_t, std::unique_ptr<strategy_type>> suit(const gameField&, fieldObjects);
        column(uint16_t, fieldObjects);
        column(fieldObjects);
        void fill(steps_type&) const override;
        bool fit(const gameField&) const override;
    private:
        uint16_t nColumn;
    };

    class diagonal : public strategy_type
    {
    public:
        enum class types { MAIN, SIDE };
        static std::pair<uint16_t, std::unique_ptr<strategy_type>> suit(const gameField&, fieldObjects);
        diagonal(types, fieldObjects);
        diagonal(fieldObjects);
        void fill(steps_type&) const override;
        bool fit(const gameField&) const override;
    private:
        types Type;
        std::array<steps_type::value_type, 3> Steps;
    };

    class random : public strategy_type
    {
    public:
        random(fieldObjects);
        void fill(steps_type&) const override;
        bool fit(const gameField&) const override;
    };

    void create_strat();
    void change_strat();

    steps_type Steps;
    std::unique_ptr<strategy_type> curStrat;
    const gameField& Arena;
    fieldObjects Side;
};