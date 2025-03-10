#pragma once

#include <vector>
#include <complex>
#include <functional>
#include <utility>

#include "FFTWAllocator.hpp"
#include "MatrixProxy.hpp"

class LightField
{
private:
    double size;

    size_t res;

    std::vector<std::complex<double>, FFTWAllocator<std::complex<double>>> field;
public:
    LightField(double size_, size_t res_) :
        size(size_),
        res(res_),
        field(res_ * res_)
    {}

    LightField& populate(std::function<std::complex<double>(double, double)> generator)
    {
        double h = size / res;

        for (size_t i = 0; i != res; ++i)
        {
            for (size_t j = 0; j != res; ++j)
            {
                double x = (i - res * 0.5) * h;
                double y = (j - res * 0.5) * h;
                field[j * res + i] = generator(x, y);
            }
        }

        return *this;
    }

    LightField& applyFilter(std::function<std::complex<double>(double, double)> filter)
    {
        double h = size / res;

        for (size_t i = 0; i != res; ++i)
        {
            for (size_t j = 0; j != res; ++j)
            {
                double x = (i - res * 0.5) * h;
                double y = (j - res * 0.5) * h;
                field[j * res + i] *= filter(x, y);
            }
        }

        return *this;
    }

    std::complex<double>& operator()(size_t x, size_t y) noexcept
    {
        return field[y * res + x];
    }

    const std::complex<double>& operator()(size_t x, size_t y) const noexcept
    {
        return field[y * res + x];
    }

    double getSize() const noexcept { return size; }

    size_t getResolution() const noexcept { return res; }

    LightField fft() const { return fft_Impl(FFTW_FORWARD); }

    LightField ifft() const { return fft_Impl(FFTW_BACKWARD); }

    std::vector<std::vector<double>> abs() const
    {
        std::vector<std::vector<double>> result(res, std::vector<double>(res));
        for (size_t i = 0; i != res; ++i)
        {
            for (size_t j = 0; j != res; ++j)
            {
                result[j][i] = std::abs(field[j * res + i]);
            }
        }
        return result;
    }

    std::vector<std::vector<double>> angle() const
    {
        std::vector<std::vector<double>> result(res, std::vector<double>(res));
        for (size_t i = 0; i != res; ++i)
        {
            for (size_t j = 0; j != res; ++j)
            {
                result[j][i] = std::arg(field[j * res + i]);
            }
        }
        return result;
    }

private:
    LightField fft_Impl(int dftDirection) const;

    fftw_complex* get() noexcept
    {
        return reinterpret_cast<fftw_complex*>(field.data());
    }

    const fftw_complex* get() const noexcept
    {
        return reinterpret_cast<const fftw_complex*>(field.data());
    }

    fftw_complex* get(size_t x, size_t y) noexcept
    {
        return reinterpret_cast<fftw_complex*>(&field[y * res + x]);
    }

    const fftw_complex* get(size_t x, size_t y) const noexcept
    {
        return reinterpret_cast<const fftw_complex*>(&field[y * res + x]);
    }

    void shift(const fftw_complex* src, size_t srcSize, 
        size_t srcStep,  fftw_complex* dst, size_t dstSize, size_t dstStep) const;

    void fit(const fftw_complex* src, size_t srcSize, 
        size_t srcStep,  fftw_complex* dst, size_t dstSize, size_t dstStep) const;

    void transpose(fftw_complex* src, size_t srcSize, size_t srcStep) const;
};