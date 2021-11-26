/*
* Header containing Labyrinth generator
* version 2.2.1
* Author: Damir Hismatov
* Github: https://github.com/Daedma
*/
#pragma once
#include <iosfwd>
#include <map>
#include <vector>
#include <random>
#include <string>

#define WIDTH_LIMIT 15ULL//smallest acceptable width
#define HEIGHT_LIMIT 15ULL//smallest acceptable height

/*
* format for a string describing the maze generation:
* the middle three letters - the value exits (Labyrinth :: exits), can have the value hor or ver;
* following the exits value, the unsigned number is the seed, read to point;
* two numbers of unsigned type and separated by 'x' after the separating dot are
* values for width and height, have values greater than WIDTH_LIMIT and HEIGHT_LIMIT, respectively.
* =========================================
* any deviation from the format throws an exception of type std :: invalid_argument,
* in this case, the labyrinth will not be generated and have an undefined state (security guarantees for exceptions are not observed)
* =========================================
* Example: Labyrinth{"hor1234567890.1000x1000"} equal Labyrinth{Labyrinth::exits::hor, 1234567890, 1000, 1000}
*/

/*
* the width and height values may not match those provided by the user,
* since there is getting rid of parity upward
* (that is, if the user provided an even value for i, then the value i + 1 will be used)
*/
class Labyrinth final//Class for random generation of labyrinth
{
public:

    using seed_type = uint64_t;
    /*
    * ver - means building a labyrinth with exits on the sides
    * hor - means building a labyrinth with exits from above and below
    */
    enum class exits//parameters to pass to constructor and function regeneration
    {
        ver, //exits on the sides
        hor //exits from above and below
    };
    enum class objects : bool
    {
        pass = false,
        wall = true
    };
    Labyrinth(size_t, size_t, exits, size_t, size_t, size_t, seed_type = std::random_device {}());//full control over all parameters (not recommended)
    Labyrinth(exits, seed_type, size_t, size_t);//exits, seed, width, height
    Labyrinth(exits, size_t, size_t);//exits, width, height
    explicit Labyrinth(const std::string&);//seed
    const seed_type& regenerate();//generate new labyrinth with new seed and old width, height, exits
    const seed_type& regenerate(const std::string&);//generate new labyrinth
    const seed_type& regenerate(exits, seed_type, size_t, size_t);//generate new labyrinth
    const seed_type& regenerate(exits, size_t, size_t);//generate new labyrinth
    void regenerate(exits, size_t, size_t, size_t, size_t, size_t, seed_type = std::random_device {}());//full control over all parameters (not recommended)
    const seed_type& seed() const//get seed
    {
        return gen_key;
    }
    void swap(Labyrinth&) noexcept;
    const std::string seed_s() const;//get seed in string perfomance
    bool is_path(size_t nRow, size_t nColumn) const
    {
        return !escape.at(nRow).at(nColumn);
    }
    std::pair<size_t, size_t> entry() const;//entry coordinates
    std::pair<size_t, size_t> exit() const;//exit coordinates
    objects at(size_t nRow, size_t nColumn) const
    {
        return static_cast<objects>(bBody.at(nRow).at(nColumn));
    }
    std::pair<size_t, size_t> size() const { return { height, width }; }

private:

    template<typename T>
    using _2dArray = std::vector<std::vector<T>>;
    //members
    struct
    {
        size_t max_branch, branch_rate, max_size;
    } branch_param;//parameter pack
    size_t width, height;
    _2dArray<bool> bBody;//main parameter describing the labyrinth
    _2dArray<bool> escape;//exit map in a bool array perfomance
    seed_type gen_key;//generation key
    std::default_random_engine engine;//random engine
    //functions
    exits init(const std::string&);//initialization of labyrinth parameters via string
    inline exits spot_ex() const;//spot labyrinth type (exits :: ver or exits :: hor)
    void init_branch_param();//initializing branch_param based on width and height
    std::multimap<size_t, size_t> init_map();//makes a map of the exit route
    void reset();//resets path and maze to initial state (sets bits to escape and bBody to true)
    void build_path(exits);//creates an exit path
    void build_subpath(exits);//full filling of the labyrinth with branches
    std::pair<size_t, size_t> subpath(int64_t, int64_t, exits);//for step 1 in building branches from path
    void subpath(int64_t, int64_t, size_t);//for building branches
    bool hor_check(size_t, size_t, size_t, short = 1) const;//function to check for cross paths when building along X-coordinate
    bool ver_check(size_t, size_t, size_t, short = 1) const;//function to check for cross paths when building along Y-coordinate
    bool h_impasse(size_t, size_t, size_t) const;//function to avoid "getting stuck" between two parallel paths when building a path along the X-coordinate
    bool v_impasse(size_t, size_t, size_t) const;//function to avoid "getting stuck" between two parallel paths when building a path along the Y-coordinate
    void resize(size_t, size_t);
};

void swap(Labyrinth&, Labyrinth&) noexcept;

/*
* default displays Labyrinth and path separately
* use macros _PRINT_WITH_EXIT_ , if you want highlight exit inside labyrinth
*/
std::ostream& operator<<(std::ostream&, const Labyrinth&);