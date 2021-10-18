#include "sum.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <fstream>
#define INDEX_N char(252) 
#define ALPHA char(224)

Sum::Sum(std::function<value_type(size_t)> member) noexcept :
    NumMember { std::move(member) }, nStart { 0ULL }, nIteration { 0ULL },
    LastMember { 0.L }, PartialSum { 0.L }, Precision { 0.L }, Print { true }{}

Sum::value_type Sum::calc(size_t from, size_t to)
{
    if (from > to) throw std::invalid_argument { "from greater than to" };
    nStart = from;
    nIteration = to;
    value_type result = 0;
    value_type curMemb, nextMemb, curPrecision;
    curMemb = NumMember(from);
    std::cout << "Current values:\n";
    while (from <= to)
    {
        nextMemb = NumMember(from + 1);
        result += curMemb;
        curPrecision = std::abs(nextMemb / result);
        if (Print)
            std::cout << "n = " << from << " a" << INDEX_N << "= " << curMemb << " S" << INDEX_N
            << " = " << result << ' ' << ALPHA << INDEX_N << " = " << curPrecision << '\n';
        curMemb = nextMemb;
        ++from;
    }
    LastMember = curMemb;
    PartialSum = result;
    Precision = curPrecision;
    return PartialSum;
}

Sum::value_type Sum::calc(size_t to)
{
    return calc(1ULL, to);
}

Sum::value_type Sum::calc(size_t from, value_type precision)
{
    if (precision <= 0) throw std::invalid_argument { "Precision is no-positive" };
    nStart = from;
    value_type result = 0;
    value_type curMemb, nextMemb, curPrecision = INFINITY;
    curMemb = NumMember(from);
    std::cout << "Current values:\n";
    while (curPrecision > precision)
    {
        nextMemb = NumMember(from + 1);
        result += curMemb;
        curPrecision = std::abs(nextMemb / result);
        if (Print)
            std::cout << "n = " << from << " a" << INDEX_N << "= " << curMemb << " S" << INDEX_N
            << " = " << result << ' ' << ALPHA << INDEX_N << " = " << curPrecision << '\n';
        curMemb = nextMemb;
        ++from;
    }
    nIteration = from;
    LastMember = curMemb;
    PartialSum = result;
    Precision = curPrecision;
    return PartialSum;
}

Sum::value_type Sum::calc(value_type precision)
{
    return calc(1ULL, precision);
}

bool Sum::is_avail() const noexcept
{
    return std::isnormal(LastMember) && std::isnormal(PartialSum) && std::isnormal(Precision);
}

std::pair<size_t, size_t> Sum::get_range() const noexcept
{
    return { nStart, nIteration };
}

Sum::value_type Sum::get_last() const noexcept
{
    return LastMember;
}

Sum::value_type Sum::get_sum() const noexcept
{
    return PartialSum;
}

Sum::value_type Sum::get_precision() const noexcept
{
    return Precision;
}

Sum::value_type Sum::calc_n(size_t n) const
{
    return NumMember(n);
}

Sum::Cache::Cache(const std::filesystem::path& Path) noexcept :
    SumCache {}, CachePath { Path }, error { false }
{
    upload();
}

void Sum::Cache::add(const cache_type::key_type& Args, const Sum& CalcSum)
{
    SumCache[Args] = { CalcSum.get_range().second, CalcSum.get_last(), CalcSum.get_sum(), CalcSum.get_precision() };
    unload();
}

bool Sum::Cache::contains(const cache_type::key_type& Args) const noexcept
{
    return SumCache.count(Args);
}

const Sum::Cache::cache_type::mapped_type& Sum::Cache::get(const cache_type::key_type& Args) const
{
    return SumCache.at(Args);
}

bool Sum::Cache::is_avail() const noexcept
{
    return !error;
}

bool Sum::Cache::edit_path(const std::filesystem::path& NewPath)
{
    CachePath = NewPath;
    error = false;
    upload();
    unload();
    return !error;
}

void Sum::Cache::upload()
{
    if (error) return;
    std::fstream fs { CachePath, std::ios::in };
    if (!fs)//если не удалось открыть файл, то создадим его
    {
        fs.close();
        fs.clear();
        fs.open(CachePath, std::ios::out);
        if (!fs)
            error = true;
        return;
    }
    else//иначе считаем из него данные
    {
        value_type x;//x
        value_type alpha;//точность вычисления
        value_type argalpha;//введдёная пользователем точность
        size_t nit;//количество просуммированных членов
        value_type lastm,//последний просуммированный член
            res;//частичная сумма
        while (fs >> std::ws && !fs.eof())//пока не достигнем конца файла
        {
            bool second_arg_is_int;
            fs >> second_arg_is_int;
            fs >> x;
            if (second_arg_is_int)
                fs >> nit;
            else
                fs >> argalpha;
            fs >> nit;
            fs >> lastm;
            fs >> res;
            fs >> alpha;
            if (!fs)//если целостность кэша нарушена
            {
                std::cerr << "Data cache is corrupted\n";
                clear_cache();
                std::cerr << "Cache file cleared\n";
                unload();
            }
            else
            {
                if (second_arg_is_int)
                    SumCache[{x, nit}] = { nit, lastm, res, alpha };
                else
                    SumCache[{x, argalpha}] = { nit, lastm, res, alpha };
            }
        }
    }
}

void Sum::Cache::unload() const noexcept
{
    if (error) return;
    std::ofstream ofs { CachePath, std::ios::out | std::ios::trunc };
    if (!ofs)
    {
        error = true;
        return;
    }
    for (const auto& i : SumCache)
    {
        const auto& [nit, lastm, res, aplha] = i.second;
        bool second_arg_is_int = std::holds_alternative<size_t>(i.first.second);
        ofs << second_arg_is_int << ' ' << i.first.first << ' ';
        if (second_arg_is_int)
            ofs << std::get<size_t>(i.first.second);
        else
            ofs << std::get<value_type>(i.first.second);
        ofs << ' ' << nit << ' ' << lastm << ' ' << res << ' ' << aplha << ' ';
    }
}

void Sum::Cache::clear_cache() const noexcept
{
    std::ofstream ofs { CachePath, std::ios::out | std::ios::trunc };
}