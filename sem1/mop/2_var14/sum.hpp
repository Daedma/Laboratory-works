#pragma once
#include <functional>
#include <filesystem>
#include <variant>
#include <tuple>
#include <map>

class Sum final
{
public:
    using value_type = long double;

    Sum() = default;
    Sum(std::function<value_type(size_t)>) noexcept;
    value_type calc(size_t, size_t);
    value_type calc(size_t);
    value_type calc(size_t, value_type);
    value_type calc(value_type);
    bool is_avail() const noexcept;
    std::pair<size_t, size_t> get_range() const noexcept;
    value_type get_last() const noexcept;
    value_type get_sum() const noexcept;
    value_type get_precision() const noexcept;
    value_type calc_n(size_t) const;

private:
    std::function<value_type(size_t)> NumMember;
    size_t nStart;
    size_t nIteration;
    value_type LastMember;
    value_type PartialSum;
    value_type Precision;
    bool Print;

public:
    class Cache final
    {
        void upload();//загрузить данные из файла кэша
        void unload() const noexcept;//выгрузить текущие данные в файл кэша
        void clear_cache() const noexcept;
    public:
        using cache_type = std::map< std::pair < value_type, std::variant <size_t, value_type>>, std::tuple<size_t, value_type, value_type, value_type>>;
        explicit Cache(const std::filesystem::path&) noexcept;
        void add(const cache_type::key_type&, const Sum&);
        bool contains(const cache_type::key_type&) const noexcept;
        const cache_type::mapped_type& get(const cache_type::key_type&) const;
        bool is_avail() const noexcept;
        bool edit_path(const std::filesystem::path&);
    private:
        cache_type SumCache;
        std::filesystem::path CachePath;
        mutable bool error;
    };
};