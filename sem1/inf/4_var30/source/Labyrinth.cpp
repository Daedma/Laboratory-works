#include "Labyrinth.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <exception>
#include <sstream>

//#define _PRINT_WITHOUT_EXIT_
//#define _PRINT_WITH_EXIT_ 

#define BLOCK char(219)//aggregate for print walls of labyrinth
#ifndef STATUS_CUP
#define STATUS_CUP 5 
/*Maximum amount of fails when generating subbranches
 *Recommended values STATUS_CUP >= 4
 *Affects the speed of construction
 */
#endif // !STATUS_CUP

#ifndef COEFFICIENT
#define COEFFICIENT 0.25
 /*As you increase, the curvature of the exit path decreases.
  *When decreasing, it increases.
  *Increase if there are problems with build time.
  */
#endif // !COEFFICIENT

#define PROC_COEFF 0.9
#define MAX_BRANCH_COEFF 0.1
#define MAX_SIZE_COEFF 0.25
#define DEFAULT_BRANCHINESS 1000ULL
#define Y_LIMIT 1
#define STEP 2//step

Labyrinth::Labyrinth(size_t _width, size_t _height, exits ex, size_t max_branch, size_t branch_rate, size_t branch_size, Labyrinth::seed_type _Seed) :
    branch_param { max_branch, branch_rate, branch_size }, width { (_width / 2 * 2 + 1) < WIDTH_LIMIT ? WIDTH_LIMIT : _width / 2 * 2 + 1 }, height { (_height / 2 * 2 + 1) < HEIGHT_LIMIT ? HEIGHT_LIMIT : _height / 2 * 2 + 1 },
    bBody(height, std::vector<bool>(width, true)), escape(height, std::vector<bool>(width, true)), gen_key(_Seed), engine(_Seed)
{
    build_path(ex);
    build_subpath(ex);
}

Labyrinth::Labyrinth(exits _Ex, seed_type _Seed, size_t _Width, size_t _Height) :
    width((_Width / 2 * 2 + 1) < WIDTH_LIMIT ? WIDTH_LIMIT : _Width / 2 * 2 + 1), height((_Height / 2 * 2 + 1) < HEIGHT_LIMIT ? HEIGHT_LIMIT : _Height / 2 * 2 + 1), gen_key(_Seed), engine(_Seed), bBody(height, std::vector<bool>(width, true)),
    escape(height, std::vector<bool>(width, true))
{
    init_branch_param();
    build_path(_Ex);
    build_subpath(_Ex);
}

Labyrinth::Labyrinth(exits _Ex, size_t _Width, size_t _Height) :
    width((_Width / 2 * 2 + 1) < WIDTH_LIMIT ? WIDTH_LIMIT : _Width / 2 * 2 + 1), height((_Height / 2 * 2 + 1) < HEIGHT_LIMIT ? HEIGHT_LIMIT : _Height / 2 * 2 + 1), gen_key(std::random_device {}()), engine(gen_key), bBody(height, std::vector<bool>(width, true)),
    escape(height, std::vector<bool>(width, true))
{
    init_branch_param();
    build_path(_Ex);
    build_subpath(_Ex);
}

Labyrinth::Labyrinth(const std::string& _Seed)
{
    exits ex = init(_Seed);
    resize(width, height);
    init_branch_param();
    build_path(ex);
    build_subpath(ex);
}

const Labyrinth::seed_type& Labyrinth::regenerate()
{
    gen_key = std::random_device {}();
    engine.seed(gen_key);
    const exits ex = spot_ex();
    reset();
    build_path(ex);
    build_subpath(ex);
    return gen_key;
}

const Labyrinth::seed_type& Labyrinth::regenerate(const std::string& _Seed)
{
    exits ex = init(_Seed);
    resize(width, height);
    init_branch_param();
    build_path(ex);
    build_subpath(ex);
    return gen_key;
}

const Labyrinth::seed_type& Labyrinth::regenerate(exits _Ex, seed_type _Seed, size_t _Width, size_t _Height)
{
    gen_key = _Seed;
    engine.seed(gen_key);
    resize(_Width, _Height);
    init_branch_param();
    build_path(_Ex);
    build_subpath(_Ex);
    return gen_key;
}

const Labyrinth::seed_type& Labyrinth::regenerate(exits _Ex, size_t _Width, size_t _Height)
{
    return regenerate(_Ex, std::random_device {}(), _Width, _Height);
}

void Labyrinth::swap(Labyrinth& rhs) noexcept
{
    using std::swap;
    swap(*this, rhs);
}

const std::string Labyrinth::seed_s() const
{
    exits ex = spot_ex();
    std::ostringstream oss;
    if (ex == exits::hor)
        oss << "hor";
    else
        oss << "ver";
    oss << gen_key << '.' << width << 'x' << height;
    return oss.str();
}

std::pair<size_t, size_t> Labyrinth::entry() const
{
    if (spot_ex() == exits::hor)
        return { static_cast<size_t>(((height / 2) % 2) ? height / 2 : height / 2 + 1), 0 };//x,y
    else
        return { 0, static_cast<size_t>(((width / 2) % 2) ? width / 2 : width / 2 + 1) };
}

std::pair<size_t, size_t> Labyrinth::exit() const
{
    if (spot_ex() == exits::hor)
    {
        for (size_t y = 0; y != height; ++y)
            if (!escape[y][width - 1])
                return { y, width - 1 };
    }
    else
    {
        for (size_t x = 0; x != width; ++x)
            if (!escape[height - 1][x])
                return { height - 1, x };
    }
    return{ 0ULL, 0ULL };
}

Labyrinth::exits Labyrinth::init(const std::string& str)
{
    exits ex;
    if (str.substr(0, 3) == "hor")
        ex = exits::hor;
    else if (str.substr(0, 3) == "ver")
        ex = exits::ver;
    else
        throw std::invalid_argument { "unexpected exits type" };

    const size_t point = str.find('.');
    if (point == std::string::npos)
        throw std::invalid_argument("format not respected");

    std::stringstream ss { str.substr(3, point) };
    if (!(ss >> gen_key))
        throw std::invalid_argument("format not respected");

    const size_t x_delimiter = str.find('x');
    if (x_delimiter == std::string::npos)
        throw std::invalid_argument("format not respected");

    ss.str(str.substr(point + 1, x_delimiter));
    if (!(ss >> width))
        throw std::invalid_argument("format not respected");
    width = (width / 2 * 2 + 1) < WIDTH_LIMIT ? WIDTH_LIMIT : width / 2 * 2 + 1;

    ss.str(str.substr(x_delimiter + 1));
    if (!(ss >> height))
        throw std::invalid_argument("format not respected");
    height = (height / 2 * 2 + 1) < HEIGHT_LIMIT ? HEIGHT_LIMIT : height / 2 * 2 + 1;
    return ex;
}

void Labyrinth::init_branch_param()
{
    branch_param = { static_cast<size_t>((width + height) / 2 * MAX_BRANCH_COEFF), DEFAULT_BRANCHINESS, static_cast<size_t>((width + height) / 2 * MAX_SIZE_COEFF) };//max_branch, branch_rate, max_size
}

std::multimap<size_t, size_t> Labyrinth::init_map()
{
    std::multimap<size_t, size_t> escape_map;
    for (size_t y = 1; y != height - 2; ++y)
        for (size_t x = 1; x != width - 2; ++x)
            if (!escape[y][x])
                escape_map.insert(std::pair<const size_t, size_t>{y, x});
    return escape_map;
}

Labyrinth::exits Labyrinth::spot_ex() const
{
    if (!bBody[(((height / 2) % 2) ? height / 2 : height / 2 + 1)][0])
        return exits::hor;
    else
        return exits::ver;
}

void Labyrinth::build_subpath(exits ex)
{
    std::multimap<size_t, size_t> escape_map = init_map();
    std::bernoulli_distribution ber(branch_param.branch_rate / 1000.);
    std::vector<std::multimap<size_t, size_t>::value_type> sample_map {};
    std::sample(escape_map.cbegin(), escape_map.cend(), std::back_inserter(sample_map), (height * width) / (height + width), engine);
#ifndef _FAST_BUILD_
    std::vector<std::pair<size_t, size_t>> ends {};
#endif // !_FAST_BUILD_
    //step 1
    for (auto& i : sample_map)
    {
        if (i.second % 2 == 1 && i.first % 2 == 1 && ber(engine))
        {
#ifdef _FAST_BUILD_
            subpath(i.second, i.first, 0, sqrt(branch_param.max_size));
#endif // _FAST_BUILD_
#ifndef _FAST_BUILD_
            ends.emplace_back(subpath(i.second, i.first, ex));
#endif // !_FAST_BUILD_
        }
    }
    //step 2
#ifndef _FAST_BUILD_
    std::shuffle(ends.begin(), ends.end(), engine);
    for (const auto& [x, y] : ends)
    {
        subpath(x, y, branch_param.max_branch);
    }
#endif // !_FAST_BUILD_
    //step 3 (final)
    if (ex == exits::hor)
        for (size_t x = 1; x != width - 1; ++x)
        {
            for (size_t y = 1; y != height - 1; ++y)
            {
                if (!bBody[y][x] && ber(engine) && x % 2 == 1 && y % 2 == 1)
                    subpath(x, y, branch_param.max_branch);
            }
        }
    else
        for (size_t y = 1; y != height - 1; ++y)
        {
            for (size_t x = 1; x != width - 1; ++x)
            {
                if (!bBody[y][x] && ber(engine) && x % 2 == 1 && y % 2 == 1)
                    subpath(x, y, branch_param.max_branch);
            }
        }
}

std::pair<size_t, size_t> Labyrinth::subpath(int64_t x, int64_t y, exits ex)
{
    std::bernoulli_distribution d(0.5);
    std::uniform_int_distribution<> d_x;
    std::uniform_int_distribution<> d_y;
    uint64_t counter = 0;
    uint8_t status = 0;
    bool flag = true;
    int dx, dy;
    if (ex == exits::hor)
    {
        d_x.param(std::uniform_int_distribution<>::param_type { -1, 1 });
        if (d(engine))
            d_y.param(std::uniform_int_distribution<>::param_type { 0, 1 });
        else
            d_y.param(std::uniform_int_distribution<>::param_type { -1, 0 });
    }
    else
    {
        d_y.param(std::uniform_int_distribution<>::param_type { -1, 1 });
        if (d(engine))
            d_x.param(std::uniform_int_distribution<>::param_type { 0, 1 });
        else
            d_x.param(std::uniform_int_distribution<>::param_type { -1, 0 });
    }
    while (counter < branch_param.max_size && flag)
    {
        dx = d_x(engine);
        dy = d_y(engine);
        if (dx && x + dx * STEP < width - 1 && x + dx * STEP > 0 && bBody[y][x + dx * STEP])
        {
            x += dx * STEP;
            bBody[y][x - dx] = bBody[y][x] = false;
            status = 0;
            ++counter;
        }
        else if (dx)
        {
            ++status;
            if (status == STATUS_CUP) flag = false;
        }
        if (dy && y + dy * STEP > 0 && y + dy * STEP < height - 1 && bBody[dy * STEP + y][x])
        {
            y += dy * STEP;
            bBody[y - dy][x] = bBody[y][x] = false;
            status = 0;
            ++counter;
        }
        else if (dy)
        {
            ++status;
            if (status == STATUS_CUP) flag = false;
        }
    }
    return std::make_pair(static_cast<size_t>(x), static_cast<size_t>(y));
}

void Labyrinth::subpath(int64_t x, int64_t y, size_t re_count)
{
    std::uniform_int_distribution<> d(-1, 1);
    std::bernoulli_distribution ber(branch_param.branch_rate / 1000.);
    uint64_t counter = 0;
    uint8_t status = 0;
    bool flag = true;
    int dx, dy;
    while (counter < branch_param.max_size && flag)
    {
        dx = d(engine), dy = d(engine);
        if (dx && x + dx * STEP < width - 1 && x + dx * STEP > 0 && bBody[y][x + dx * STEP])
        {
            x += dx * STEP;
            bBody[y][x - dx] = bBody[y][x] = false;
            status = 0;
            ++counter;
        }
        else if (dx)
        {
            ++status;
            if (status == STATUS_CUP) flag = false;
        }
        if (dy && y + dy * STEP > 0 && y + dy * STEP < height - 1 && bBody[dy * STEP + y][x])
        {
            y += dy * STEP;
            bBody[y - dy][x] = bBody[y][x] = false;
            status = 0;
            ++counter;
        }
        else if (dy)
        {
            ++status;
            if (status == STATUS_CUP) flag = false;
        }
        if (ber(engine) && re_count && !status)
        {
            subpath(x, y, re_count - 1);
        }
    }
}

bool Labyrinth::hor_check(size_t indx, size_t indy, size_t max_x, short course) const
{
    for (; indx <= max_x; indx += course)
        if (!bBody[indy][indx])
            return true;
    return false;
}

bool Labyrinth::ver_check(size_t indx, size_t indy, size_t max_y, short course) const
{
    for (; indy <= max_y; indy += course)
        if (!bBody[indy][indx])
            return true;
    return false;
}

bool Labyrinth::h_impasse(size_t x, size_t y, size_t max_x) const
{
    if (y < 5) return false;
    if (y > height - 6) return false;
    return hor_check(x, y + 2, max_x, 2) && hor_check(x, y - 2, max_x, 2);
}

bool Labyrinth::v_impasse(size_t x, size_t y, size_t max_y) const
{
    if (x < 5) return false;
    if (x > width - 6) return false;
    return ver_check(x, y + 2, max_y, 2) && ver_check(x, y - 2, max_y, 2);
}

void Labyrinth::reset()
{
    for (auto& y : bBody)
        for (auto x : y)
            x = true;
    for (auto& y : escape)
        for (auto x : y)
            x = true;
}

void Labyrinth::resize(size_t NewWidth, size_t NewHeight)
{
    width = (NewWidth / 2 * 2 + 1) < WIDTH_LIMIT ? WIDTH_LIMIT : NewWidth / 2 * 2 + 1;
    height = (NewHeight / 2 * 2 + 1) < HEIGHT_LIMIT ? HEIGHT_LIMIT : NewHeight / 2 * 2 + 1;
    bBody.clear();
    bBody.resize(height, std::vector<bool>(width, true));
    escape.clear();
    escape.resize(height, std::vector<bool>(width, true));
}

void Labyrinth::build_path(exits ex)
{
    std::uniform_int_distribution<> delta(-1, 1);
    std::bernoulli_distribution proc(PROC_COEFF);
    int64_t x, y;//coordinates
    bool status;//proc
    int dy;//dy - delta y
    int dx;//dx - delta x
    bool no_impasse = true;
    if (ex == exits::hor)
    {
        std::bernoulli_distribution cf(std::clamp(1.56 - pow(static_cast<double>(height) / width, COEFFICIENT), 0.1, 1.));//cf - curvature factor
        int64_t max_x = 0;
        x = 0, y = ((height / 2) % 2) ? height / 2 : height / 2 + 1;
        escape[y][x + 1] = bBody[y][x + 1] = escape[y][x] = bBody[y][x] = false;
        x += 1;
        while (x < width - 3)
        {
            no_impasse = true;
            status = proc(engine);
            dx = delta(engine);
            if (dx == -1 && cf(engine) || x < 3 || y < 5 || y > height - 6)
                dx = 1;
            dy = delta(engine);
            if (status && dx && x + dx * 2 > 0 && escape[y][x + dx * 2] && ((dx == -1) ? (no_impasse = !h_impasse(x + dx * STEP, y, max_x)) : true))
            {
                x += dx * 2;
                escape[y][x - dx] = bBody[y][x - dx] =
                    escape[y][x] = bBody[y][x] = false;
                if (proc(engine) && x + dx * 2 > 0 && x + dx * 2 < width && escape[y][x + dx * 2] && ((dx == -1) ? !h_impasse(x + dx * STEP, y, max_x) : true))
                {
                    x += dx * 2;
                    escape[y][x - dx] = bBody[y][x - dx] =
                        escape[y][x] = bBody[y][x] = false;
                }
            }
            else if (status && dx == -1 && !no_impasse)
            {
                dx = 1;
                if (x + dx * 2 > 0 && escape[y][x + dx * 2])
                {
                    x += dx * 2;
                    escape[y][x - dx] = bBody[y][x - dx] =
                        escape[y][x] = bBody[y][x] = false;
                    if (proc(engine) && x + dx * 2 > 0 && x + dx * 2 < width && escape[y][x + dx * 2])
                    {
                        x += dx * 2;
                        escape[y][x - dx] = bBody[y][x - dx] =
                            escape[y][x] = bBody[y][x] = false;
                    }
                }
            }
            max_x = std::max(max_x, x);
            if (dy && y + dy * STEP > 0 && y + dy * STEP < height && escape[y + dy * STEP][x] && !hor_check(x, y + dy * STEP, max_x, 2))
            {
                y += dy * STEP;
                escape[y - dy][x] = bBody[y - dy][x] = escape[y][x] = bBody[y][x] = false;
                if (!cf(engine) && y + dy * STEP > 0 && y + dy * STEP < height && escape[y + dy * STEP][x] && !hor_check(x, y + dy * STEP, max_x, 2))
                {
                    y += dy * STEP;
                    escape[y - dy][x] = bBody[y - dy][x] = escape[y][x] = bBody[y][x] = false;
                }
            }
            else if (dy)
            {
                dy = -dy;
                if (y + dy * STEP > 0 && y + dy * STEP < height && escape[y + dy * STEP][x] && !hor_check(x, y + dy * STEP, max_x, 2))
                {
                    y += dy * STEP;
                    escape[y - dy][x] = bBody[y - dy][x] = escape[y][x] = bBody[y][x] = false;
                    if (!cf(engine) && y + dy * STEP > 0 && y + dy * STEP < height && escape[y + dy * STEP][x] && !hor_check(x, y + dy * STEP, max_x, 2))
                    {
                        y += dy * STEP;
                        escape[y - dy][x] = bBody[y - dy][x] = escape[y][x] = bBody[y][x] = false;
                    }
                }
            }
        }
        escape[y][x + 1] = bBody[y][x + 1] = false;
    }
    else
    {
        std::bernoulli_distribution cf(std::clamp(1.56 - pow(static_cast<double>(width) / height, COEFFICIENT), 0.1, 1.));
        int64_t max_y = 0;
        x = ((width / 2) % 2) ? width / 2 : width / 2 + 1, y = 0;
        escape[y + 1][x] = bBody[y + 1][x] = escape[y][x] = bBody[y][x] = false;
        y += 1;
        while (y < height - 3)
        {
            no_impasse = true;
            status = proc(engine);
            dy = delta(engine);
            if (dy == -1 && cf(engine) || y < 3 || x < 5 || x > width - 6)
                dy = 1;
            dx = delta(engine);
            if (status && dy && y + dy * 2 > 0 && escape[y + dy * 2][x] && ((dx == -1) ? no_impasse = !v_impasse(x, y + dy * STEP, max_y) : true))
            {
                y += dy * 2;
                escape[y - dy][x] = bBody[y - dy][x] =
                    escape[y][x] = bBody[y][x] = false;
                if (proc(engine) && y + dy * 2 > 0 && y + dy * 2 < height && escape[y + dy * 2][x] && ((dx == -1) ? !v_impasse(x, y + dy * STEP, max_y) : true))
                {
                    y += dy * 2;
                    escape[y - dy][x] = bBody[y - dy][x] =
                        escape[y][x] = bBody[y][x] = false;
                }
            }
            else if (status && dy == -1 && !no_impasse)
            {
                dy = 1;
                if (y + dy * 2 > 0 && escape[y + dy * 2][x])
                {
                    y += dy * 2;
                    escape[y - dy][x] = bBody[y - dy][x] =
                        escape[y][x] = bBody[y][x] = false;
                    if (proc(engine) && y + dy * 2 > 0 && y + dy * 2 < height && escape[y + dy * 2][x])
                    {
                        y += dy * 2;
                        escape[y - dy][x] = bBody[y - dy][x] =
                            escape[y][x] = bBody[y][x] = false;
                    }
                }
            }
            max_y = std::max(max_y, y);
            if (dx && x + dx * STEP > 0 && x + dx * STEP < width && escape[y][x + dx * STEP] && !ver_check(x + dx * STEP, y, max_y, 2))
            {
                x += dx * STEP;
                escape[y][x - dx] = bBody[y][x - dx] = escape[y][x] = bBody[y][x] = false;
                if (!cf(engine) && x + dx * STEP > 0 && x + dx * STEP < width && escape[y][x + dx * STEP] && !ver_check(x + dx * STEP, y, max_y, 2))
                {
                    x += dx * STEP;
                    escape[y][x - dx] = bBody[y][x - dx] = escape[y][x] = bBody[y][x] = false;
                }
            }
            else if (dx)
            {
                dx = -dx;
                if (x + dx * STEP > 0 && x + dx * STEP < width && escape[y][x + dx * STEP] && !ver_check(x + dx * STEP, y, max_y, 2))
                {
                    x += dx * STEP;
                    escape[y][x - dx] = bBody[y][x - dx] = escape[y][x] = bBody[y][x] = false;
                    if (!cf(engine) && x + dx * STEP > 0 && x + dx * STEP < width && escape[y][x + dx * STEP] && !ver_check(x + dx * STEP, y, max_y, 2))
                    {
                        x += dx * STEP;
                        escape[y][x - dx] = bBody[y][x - dx] = escape[y][x] = bBody[y][x] = false;
                    }
                }
            }
        }
        escape[y + 1][x] = bBody[y + 1][x] = false;
    }
}

void Labyrinth::regenerate(exits ex, size_t _width, size_t _height, size_t max_branch, size_t branch_rate, size_t branch_size, Labyrinth::seed_type _Seed)
{
    gen_key = _Seed;
    branch_param.max_branch = max_branch;
    branch_param.branch_rate = branch_rate;
    branch_param.max_size = branch_size;
    engine.seed(gen_key);
    resize(_width, _height);
    build_path(ex);
    build_subpath(ex);
}

void swap(Labyrinth& lhs, Labyrinth& rhs) noexcept
{
    lhs.swap(rhs);
}

std::ostream& operator<<(std::ostream& os, const Labyrinth& lab)
{
    for (auto y = 0; y != lab.size().first; ++y)
    {
        for (auto x = 0; x != lab.size().second; ++x)
        {
            if (lab.at(y, x) == Labyrinth::objects::wall)
                os << BLOCK;
#ifdef _PRINT_WITH_EXIT_
            else if (lab.is_path(y, x))
                os << char(177);
#endif // _PRINT_WITH_EXIT
            else
                //os << ' ';
                os << char(176);
        }
        os << std::endl;
    }
    return os;
}
