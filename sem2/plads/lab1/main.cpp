#include <fstream>

size_t count(const char* aFile)//Number of items in the file
{
    std::ifstream ifs { aFile };
    size_t num = 0;
    double d;
    while (ifs >> d) ++num;
    return num;
}

int main()
{
    size_t nfile_items = count("input.txt");
    double* items = new double[nfile_items];
    std::ifstream ifs { "input.txt" };
    for (size_t i = 0; i != nfile_items; ++i)
        ifs >> items[i];
    ifs.close();
    std::ofstream ofs { "output.txt" };
    ofs << nfile_items;
    ofs.close();
    delete[] items;
}