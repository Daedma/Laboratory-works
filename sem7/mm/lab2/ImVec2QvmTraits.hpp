#pragma once
#include <boost/qvm/vec.hpp>
#include <boost/qvm/vec_traits.hpp>
#include <imgui.h>

// Специализация vec_traits для ImVec2
namespace boost
{
	namespace qvm
	{
		template <>
		struct vec_traits<ImVec2>
		{
			typedef float scalar_type;
			static int const dim = 2;

			template <int I>
			static inline scalar_type& write_element(ImVec2& v)
			{
				return (&v.x)[I];
			}

			static scalar_type& write_element_idx(int i, ImVec2& v)
			{
				return (&v.x)[i];
			}

			template <int I>
			static inline scalar_type read_element(ImVec2 const& v)
			{
				return (&v.x)[I];
			}

			static scalar_type read_element_idx(int i, ImVec2 const& v)
			{
				return (&v.x)[i];
			}
		};

		template <>
		struct is_vec<ImVec2>
		{
			static bool const value = true;
		};
	}
}