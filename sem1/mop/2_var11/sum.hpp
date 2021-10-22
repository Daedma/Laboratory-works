#pragma once
#include <functional>
#include <filesystem>
#include <variant>
#include <tuple>
#include <map>

class Sum final
{
public:
    using value_type = long double;//тип членов последовательности

    Sum(std::function<value_type(size_t)>) noexcept;//инициализация функцией, по которой можно подсчитать члены последовательности по их номерам
    value_type calc(size_t, size_t);//подсчитать частичную сумму при n принадлежит [from, to]
    value_type calc(size_t);//эквивалентно calc(1, to)
    value_type calc(size_t, value_type);//подсчитать частичную сумму, пока не будет достигнута определенная точность
    value_type calc(value_type);//эквивалентно calc(1, precision)
    bool is_avail() const noexcept;//проверяет, нет ли среди полей бесконечностей или не чисел
    std::pair<size_t, size_t> get_range() const noexcept;//получить диапозон n, в котором производились вычисления
    value_type get_last() const noexcept;//получить последний просуммированный член 
    value_type get_sum() const noexcept;//получить частичную сумму
    value_type get_precision() const noexcept;//получить точность вычисления
    value_type calc_n(size_t) const;//найти n-ый член последовательности

private:
    std::function<value_type(size_t)> NumMember;//функцией, по которой можно подсчитать члены последовательности по их номерам
    size_t nStart;//начальное n
    size_t nIteration;//кол-во итераций
    value_type LastMember;//последний просуммированный член
    value_type PartialSum;//частичная суммф
    value_type Precision;//точность вычисления

public:
    class Cache final
    {
        void upload();//загрузить данные из файла кэша
        void unload() const noexcept;//выгрузить текущие данные в файл кэша
        void clear_cache() const noexcept;//очистить файл кэша
    public:
        //тип, в котором будут хранится сохраненные результаты
        using cache_type = std::map< std::pair < value_type, std::variant <size_t, value_type>>, std::tuple<size_t, value_type, value_type, value_type>>;
        explicit Cache(const std::filesystem::path&) noexcept;//инициализация путем к файлу, в котором будут сохранены результаты
        void add(const cache_type::key_type&, const Sum&);//загрузить результаты в кэш
        bool contains(const cache_type::key_type&) const noexcept;//проверяет, содержит ли кэш результаты с для таких параметров
        const cache_type::mapped_type& get(const cache_type::key_type&) const;//взять данные из кэша
        bool is_avail() const noexcept;//проверяет, находится ли кэш в валидном состоянии
        bool edit_path(const std::filesystem::path&);//изменить путь к файлу кэша

    private:
        cache_type SumCache;//содержит результаты вычислений
        std::filesystem::path CachePath;//путь к файлу кэша
        mutable bool error;//содержит состояние об ошибке
    };
};