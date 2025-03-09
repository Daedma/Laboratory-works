#pragma once

#include <cstdint>
#include <functional>

template<typename T, typename ReturnType = T>
class MatrixProxy
{
	class RowProxy
	{
	public:
		RowProxy(const T* data_, size_t width_, std::function<ReturnType(const T&)> mapper_) :
			data(data_), width(width_), mapper(mapper_)
		{}

		ReturnType operator[](size_t index) const
		{
			return mapper(data[index]);
		};

	private:
		const T* data;
		size_t width;
		std::function<ReturnType(const T&)> mapper;
	};

public:
	MatrixProxy(const T* data_, size_t width_, std::function<ReturnType(const T&)> mapper_) :
		data(data_), width(width_), mapper(mapper_)
	{}

	RowProxy operator[](size_t index) const
	{
		return RowProxy(data + index * width, width, mapper);
	}

private:
	const T* data;
	size_t width;
	std::function<ReturnType(const T&)> mapper;
};