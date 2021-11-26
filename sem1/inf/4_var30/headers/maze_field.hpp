#pragma once
#include <memory>
#include <vector>
#include <filesystem>
#include <utility>

class MazeObject;

class MazeField
{
public:
    MazeField(const std::filesystem::path& aPath);
    MazeField(size_t aWidth, size_t aHeight, const std::filesystem::path& aOut);
    ~MazeField();
    MazeField(MazeField&&);
    MazeField& operator=(MazeField&&);

    std::pair<size_t, size_t> size() const noexcept;
    MazeObject* get(size_t nRow, size_t nCol) { return field.at(nRow).at(nCol).get(); }
    const MazeObject* get(size_t nRow, size_t nCol) const
    { return field.at(nRow).at(nCol).get(); }
    void clear(size_t nRow, size_t nCol);
    std::pair<bool, std::pair<size_t, size_t>> rand_pass() const noexcept;
private:
    std::vector<std::vector<std::unique_ptr<MazeObject>>> field;

    std::unique_ptr<MazeObject> create_obj(char aObjSym) const;
    std::unique_ptr<MazeObject> rand_obj() const;
};