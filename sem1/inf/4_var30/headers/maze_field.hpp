#pragma once
#include <memory>
#include <vector>
#include <filesystem>
#include <utility>
#include "..\headers\maze_objects.hpp"


class MazeField
{
public:
    explicit MazeField(const std::filesystem::path& aPath);
    MazeField(size_t aWidth, size_t aHeight, const std::filesystem::path& aOut);
    ~MazeField();
    MazeField(MazeField&&);
    MazeField& operator=(MazeField&&);

    std::pair<size_t, size_t> size() const noexcept
    { return { field.size(), field[0].size() }; }

    MazeObject* get(size_t nRow, size_t nCol) { return field.at(nRow).at(nCol).get(); }

    const MazeObject* get(size_t nRow, size_t nCol) const
    { return field.at(nRow).at(nCol).get(); }

    void clear(size_t nRow, size_t nCol)
    { field.at(nRow).at(nCol).reset(new Pass); }

    std::pair<bool, std::pair<size_t, size_t>> rand_pass() const noexcept;
private:
    std::vector<std::vector<std::unique_ptr<MazeObject>>> field;

    std::unique_ptr<MazeObject> create_obj(char aObjSym) const;
    std::unique_ptr<MazeObject> rand_obj() const;
};