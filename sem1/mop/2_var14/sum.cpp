#include "sum.hpp"
#include "table_print.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <initializer_list>

Sum::Sum(std::function<value_type(size_t)> member) noexcept :
    NumMember { std::move(member) }, nStart { 0ULL }, nIteration { 0ULL },
    LastMember { 0.L }, PartialSum { 0.L }, Precision { 0.L }{}

Sum::value_type Sum::calc(size_t from, size_t to)
{
    if (from > to) throw std::invalid_argument { "from greater than to" };
    nStart = from;
    nIteration = to - from + 1;
    value_type result = 0;//текущая частичная сумма
    value_type curMemb,//текущий член последовательности
        nextMemb,//следующий член последовательности
        curPrecision = 0;//текущая точность вычислений
    curMemb = NumMember(from);//подсчитаем первый член последовательности
    std::cout << "Current values:\n";
    print_head({ { "n", 4 }, { "an", 25 }, { "Sn", 25 }, { "ALPHAn", 25 } });
    while (from <= to)
    {
        nextMemb = NumMember(from + 1);//найдём следующий член последовательности
        result += curMemb;//прибавим текущий член к частичной сумме
        if (result)
        {
            curPrecision = std::abs(nextMemb / result);//вычислим текущую точность
        }
        //выведем текущие значения
        print_line({ { from, 4 }, { curMemb, 25 }, { result, 25 }, { curPrecision, 25 } });
        if (from != to)//для последней итерации не выполнять
            curMemb = nextMemb;
        ++from;
    }
    print_end({ 4, 25, 25, 25 });
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
    value_type result = 0;//текущая частичная сумма
    value_type curMemb,//текущий член последовательности
        nextMemb,//следующий член последовательности
        curPrecision = INFINITY;//текущая точность вычислений
    curMemb = NumMember(from);//подсчитаем первый член последовательности
    std::cout << "Current values:\n";
    print_head({ { "n", 4 }, { "an", 25 }, { "Sn", 25 }, { "ALPHAn", 25 } });
    while (curPrecision > precision)//пока текущая точность больше заданной
    {
        nextMemb = NumMember(from + 1);//найдём следующий член последовательности
        result += curMemb;//прибавим текущий член к частичной сумме
        if (result)
        {
            curPrecision = std::abs(nextMemb / result);//вычислим текущую точность
        }
        //выведем текущие значения
        print_line({ { from, 4 }, { curMemb, 25 }, { result, 25 }, { curPrecision, 25 } });
        if (curPrecision > precision)//для последней итерации не выполнять
            curMemb = nextMemb;
        ++from;
    }
    print_end({ 4, 25, 25, 25 });
    nIteration = from - nStart;
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
    upload();//загрузим данные из файла кэша
}

void Sum::Cache::add(const cache_type::key_type& Args, const Sum& CalcSum)
{
    SumCache[Args] = { CalcSum.get_range().second, CalcSum.get_last(), CalcSum.get_sum(), CalcSum.get_precision() };
    unload();//выгрузим данные в кэш файл
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
        if (!fs)//если не удалось создать файл
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
        fs >> std::setprecision(16);
        while (fs >> std::ws && !fs.eof())//пока не достигнем конца файла
        {
            //в начале стоит параметр, указывающий тип второго аргумента
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
    if (!ofs)//если не удалось открыть или создать файл
    {
        error = true;
        return;
    }
    ofs << std::setprecision(16);
    for (const auto& i : SumCache)
    {
        const auto& [nit, lastm, res, aplha] = i.second;
        bool second_arg_is_int = std::holds_alternative<size_t>(i.first.second);
        ofs << second_arg_is_int << ' ' << i.first.first << ' ';//в начале ставим единицу для того, чтобы узнать тип второго аргумента
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