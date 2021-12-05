#pragma once
#include <vector>
#include <string>
#include <map>
#include "working_shedule.hpp"

class Employee
{
public:
    Employee(const std::string& aName) :
        name { aName } {}
    uint64_t month_hours_worked(Date::Months aMonth) const noexcept;
    uint64_t year_hours_worked() const noexcept;
    void add_day(const Work_schedule& aDay) { timetable[aDay.get_date().get_month()].emplace_back(aDay); }
    void add_day(Work_schedule&& aDay)
    { timetable[aDay.get_date().get_month()].emplace_back(std::move(aDay)); }
    const std::string& get_name() const noexcept { return name; }
private:
    std::string name;
    std::map<Date::Months, std::vector<Work_schedule>> timetable;
};

std::istream& operator>>(std::istream& is, Employee& aEmp);
std::ostream& operator<<(std::ostream& os, const Employee& aEmp);