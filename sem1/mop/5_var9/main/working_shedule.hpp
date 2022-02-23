#pragma once
#include <iosfwd>
#include <cstdint>
#include <string_view>

class Date
{
public:
    enum class Months : uint16_t { jan = 1, feb, apr, mar, may, jun, jul, aug, sep, oct, nov, dec };
    Date() noexcept;
    Date(const Date& rhs) noexcept :
        day { rhs.day }, month { rhs.month }{}
    Date(Date&& rhs) noexcept :
        day { std::move(rhs.day) }, month { std::move(rhs.month) }{}
    Date& operator=(const Date& rhs) noexcept
    {
        day = rhs.day;
        month = rhs.month;
        return *this;
    }
    Date& operator=(Date&& rhs) noexcept
    {
        day = std::move(rhs.day);
        month = std::move(rhs.month);
        return *this;
    }
    ~Date() {}
    explicit Date(std::string_view aDate);
    uint16_t& set_day(uint16_t aDay);
    Months& set_month(Months aMonth);
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
    Time(const Time& rhs) noexcept :
        hour { rhs.hour }, minute { rhs.minute }{}
    Time(Time&& rhs) noexcept :
        hour { std::move(rhs.hour) }, minute { std::move(rhs.minute) }{}
    Time& operator=(const Time& rhs) noexcept
    {
        hour = rhs.hour;
        minute = rhs.minute;
        return *this;
    }
    Time& operator=(Time&& rhs) noexcept
    {
        hour = std::move(rhs.hour);
        minute = std::move(rhs.minute);
        return *this;
    }
    ~Time() {}
    explicit Time(std::string_view aTime);
    Time(uint16_t aHour, uint16_t aMin);
    Time operator-(const Time& rhs) const noexcept;
    uint16_t& set_min(uint16_t aMinute);
    uint16_t& set_hour(uint16_t aHour);
    uint16_t get_min() const noexcept { return minute; }
    uint16_t get_hour() const noexcept { return hour; }
private:
    uint16_t hour;
    uint16_t minute;
};

class Work_schedule
{
public:
    Work_schedule() = default;
    Work_schedule(const Date& aDate, const Time& aArr, const Time& aLeave) :
        date { aDate }, arrival { aArr }, leaving { aLeave }{}
    Work_schedule(const Work_schedule& rhs) noexcept :
        date { rhs.date }, arrival { rhs.arrival }, leaving { rhs.leaving }{}
    Work_schedule(Work_schedule&& rhs) noexcept :
        date { std::move(rhs.date) }, arrival(std::move(rhs.arrival)), leaving { std::move(rhs.leaving) }{}
    Work_schedule& operator=(const Work_schedule& rhs) noexcept
    {
        date = rhs.date;
        arrival = rhs.arrival;
        leaving = rhs.leaving;
        return *this;
    }
    Work_schedule& operator=(Work_schedule&& rhs) noexcept
    {
        date = std::move(rhs.date);
        arrival = std::move(rhs.arrival);
        leaving = std::move(rhs.leaving);
        return *this;
    }
    ~Work_schedule() {}
    Time working_hours() const noexcept { return leaving - arrival; }
    Date& set_date(const Date& aDate) noexcept { return date = aDate; }
    Time& set_arrival(const Time& aTime) noexcept { return arrival = aTime; }
    Time& set_leaving(const Time& aTime) noexcept { return leaving = aTime; }
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