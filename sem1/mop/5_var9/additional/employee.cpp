#include "employee.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>

uint64_t Employee::month_hours_worked(Date::Months aMonth) const noexcept
{
    if (!timetable.count(aMonth))
        return 0;
    uint64_t cHours = 0, cMin = 0;
    for (const auto& i : timetable.at(aMonth))
    {
        cHours += i.working_hours().get_hour();
        cMin += i.working_hours().get_min();
    }
    return cHours + cMin / 60;
}

uint64_t Employee::year_hours_worked() const noexcept
{
    uint64_t cHour = 0;
    for (uint16_t i = 1; i <= 12; ++i)
        cHour += month_hours_worked(static_cast<Date::Months>(i));
    return cHour;
}


std::istream& operator>>(std::istream& is, Employee& rhs)
{
    std::string name;
    if (!(is >> name)) return is;
    if (!std::all_of(name.cbegin(), name.cend(), std::isalpha))
    {
        for (auto i = name.crbegin(); i != name.crend(); ++i)
            is.putback(*i);
        is.setstate(std::ios::failbit);
        return is;
    }
    Employee emp { name };
    Work_schedule cur_shed;
    while (is >> cur_shed) emp.add_day(std::move(cur_shed));
    is.clear();
    rhs = std::move(emp);
    return is;
}

std::ostream& operator<<(std::ostream& os, const Employee& rhs)
{
    return os << "Number of hours worked of " << rhs.get_name() << ":\n"
        << "January   - " << rhs.month_hours_worked(Date::Months::jan) << '\n'
        << "February  - " << rhs.month_hours_worked(Date::Months::feb) << '\n'
        << "Marth     - " << rhs.month_hours_worked(Date::Months::mar) << '\n'
        << "April     - " << rhs.month_hours_worked(Date::Months::apr) << '\n'
        << "May       - " << rhs.month_hours_worked(Date::Months::may) << '\n'
        << "June      - " << rhs.month_hours_worked(Date::Months::jun) << '\n'
        << "Jule      - " << rhs.month_hours_worked(Date::Months::jul) << '\n'
        << "August    - " << rhs.month_hours_worked(Date::Months::aug) << '\n'
        << "September - " << rhs.month_hours_worked(Date::Months::sep) << '\n'
        << "October   - " << rhs.month_hours_worked(Date::Months::oct) << '\n'
        << "November  - " << rhs.month_hours_worked(Date::Months::nov) << '\n'
        << "December  - " << rhs.month_hours_worked(Date::Months::dec) << '\n'
        << "============" << '\n'
        << "Total     - " << rhs.year_hours_worked();
}