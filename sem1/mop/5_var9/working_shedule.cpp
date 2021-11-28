#include "working_shedule.hpp"
#include <iostream>
#include <stdexcept>
#include <map>
#include <cmath>
#include <cctype>

namespace{
    inline bool is_format(std::string_view aDate, char aDelim) noexcept
    {
        return aDate.size() == 5 && std::isdigit(aDate[0]) && std::isdigit(aDate[1])
            && aDate[2] == aDelim && std::isdigit(aDate[3]) && std::isdigit(aDate[4]);
    }

    bool day_exist(uint16_t aDay, Date::Months aMonth) noexcept
    {
        static const std::map<Date::Months, uint16_t> months
        {
            { Date::Months::jan, 31 },
            { Date::Months::feb, 29 },
            { Date::Months::apr, 31 },
            { Date::Months::mar, 30 },
            { Date::Months::may, 31 },
            { Date::Months::jun, 30 },
            { Date::Months::jul, 31 },
            { Date::Months::aug, 31 },
            { Date::Months::sep, 30 },
            { Date::Months::oct, 31 },
            { Date::Months::nov, 30 },
            { Date::Months::dec, 31 }
        };
        return aDay && aDay <= months.at(aMonth);
    }
}

Date::Date() noexcept :
    day { 1 }, month { Months::jan }{}

Date::Date(std::string_view aDate)
{
    if (!is_format(aDate, '.'))
        throw std::invalid_argument { "Date format is not respected" };
    day = (aDate[0] - '0') * 10 + (aDate[1] - '0');
    uint16_t aMonth = (aDate[3] - '0') * 10 + (aDate[4] - '0');
    if (aMonth > 12 || aMonth < 1)
        throw std::invalid_argument { "Months with such a number does not exist" };
    month = static_cast<Months>(aMonth);
    if (!day_exist(day, month))
        throw std::invalid_argument { "There is no such day in a month" };
}

Time::Time(std::string_view aTime)
{
    if (!is_format(aTime, ':'))
        throw std::invalid_argument { "Date format is not respected" };
    hour = (aTime[0] - '0') * 10 + (aTime[1] - '0');
    if (hour > 23)
        throw std::invalid_argument { "The number of hours should not exceed 23" };
    minute = (aTime[3] - '0') * 10 + (aTime[5] - '0');
    if (minute > 59)
        throw std::invalid_argument { "The number of minutes should not exceed 59" };
}

Time::Time(uint16_t aMin, uint16_t aHour) :
    minute { aMin }, hour { aHour }
{
    if (hour > 23)
        throw std::invalid_argument { "The number of hours should not exceed 23" };
    if (minute > 59)
        throw std::invalid_argument { "The number of minutes should not exceed 59" };
}

Time Time::operator-(const Time& rhs) const noexcept
{
    int16_t diff = hour * 60 + minute - rhs.hour * 60 - rhs.minute;
    if (diff < 0)
        return Time { 24 + diff / 60, 60 + diff % 60 };
    return Time { diff / 60, diff % 60 };
}

std::istream& operator>>(std::istream& is, Date& rhs)
{
    std::string tmp;
    is >> tmp;
    rhs = Date { tmp };
    return is;
}

std::istream& operator>>(std::istream& is, Time& rhs)
{
    std::string tmp;
    is >> tmp;
    rhs = Time { tmp };
    return is;
}

std::istream& operator>>(std::istream& is, Work_schedule& rhs)
{
    Date d;
    Time at, lt;
    is >> d >> at >> lt;
    rhs = Work_schedule { d, at, lt };
    return is;
}

std::ostream& operator<<(std::ostream& os, const Date& rhs)
{
    if (rhs.get_day() < 10)
        os << '0';
    os << rhs.get_day() << '.';
    if (static_cast<uint16_t>(rhs.get_month()) < 10)
        os << '0';
    os << static_cast<uint16_t>(rhs.get_month());
    return os;
}


std::ostream& operator<<(std::ostream& os, const Time& rhs)
{
    if (rhs.get_hour() < 0)
        os << '0';
    os << rhs.get_hour() << ':';
    if (rhs.get_min() < 0)
        os << '0';
    os << rhs.get_min();
    return os;
}

std::ostream& operator<<(std::ostream& os, const Work_schedule& rhs)
{
    return os << rhs.get_date() << '\n' << rhs.get_arrival() << '\n' << rhs.get_leaving();
}