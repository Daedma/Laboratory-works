#pragma once
#include <iosfwd>
#include <cstdint>
#include <string_view>

class Date
{
public:
    enum class Months : uint16_t { jan = 1, feb, apr, mar, may, jun, jul, aug, sep, oct, nov, dec };
    Date() noexcept;
    explicit Date(std::string_view aDate);
    uint16_t get_day() const noexcept { return day; }
    Months get_month() const noexcept { return month; }
private:
    uint16_t day;
    Months month;
};

class Time
{
public:
    Time() = default;
    explicit Time(std::string_view aTime);
    Time(uint16_t aMin, uint16_t aHour);
    Time operator-(const Time& rhs) const noexcept;
    uint16_t get_min() const noexcept { return minute; }
    uint16_t get_hour() const noexcept { return hour; }
private:
    uint16_t minute;
    uint16_t hour;
};

class Work_schedule
{
public:
    Work_schedule() = default;
    Work_schedule(const Date& aDate, const Time& aArr, const Time& aLeave) :
        date { aDate }, arrival { aArr }, leaving { aLeave }{}
    Time working_hours() const noexcept { return leaving - arrival; }
    const Date& get_date() const noexcept { return date; }
    const Time& get_arrival() const noexcept { return arrival; }
    const Time& get_leaving() const noexcept { return leaving; }
private:
    Date date;
    Time arrival;
    Time leaving;
};

std::istream& operator>>(std::istream& is, Date& rhs);
std::istream& operator>>(std::istream& is, Time& rhs);
std::istream& operator>>(std::istream& is, Work_schedule& rhs);

std::ostream& operator<<(std::ostream& os, const Date& rhs);
std::ostream& operator<<(std::ostream& os, const Time& rhs);
std::ostream& operator<<(std::ostream& os, const Work_schedule& rhs);