#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "neural_nets\detail\matrix_utils.h"

namespace neural_nets
{
	namespace detail
	{
		namespace math_utils
		{
			template <typename iter>
			auto mean(iter begin_, iter end_) -> typename std::remove_reference<decltype(*begin_)>::type
			{
				using T = decltype(mean(begin_, end_));
				T tmp(0);
				size_t cnt = 0;
				for (auto it = begin_; it != end_; ++it, ++cnt) {
					tmp += *it;
				}
				return tmp / cnt;
			}

			template <typename container_type>
			auto mean(container_type const &container) -> typename std::remove_reference<decltype(*std::begin(container))>::type
			{
				return mean(std::begin(container), std::end(container));
			}

			template <typename T>
			T normalized_error(boost::numeric::ublas::matrix<T> const &first_, boost::numeric::ublas::matrix<T> const &second_)
			{
				T err(0);
				size_t cnt = 1;
				for (size_t i = 0; i < first_.size1(); ++i, ++cnt) {
					T tmp = first_(i, 0) - second_(i, 0);
					err += tmp*tmp;
				}
				return err / cnt;
			}


			template <typename T>
			T calc_optimal_epsilon(T const &x_)
			{
				return std::max(T(1), std::abs(x_))*std::sqrt(std::numeric_limits<T>::epsilon());
			}

			template<typename iter>
			auto maximum_change(iter begin_, iter end_) -> typename std::remove_reference<decltype(*begin_)>::type
			{
				using T = decltype(maximum_change(begin_, end_));
				T max_change(0);
				for (auto it = begin_; it != end_; ++it) {
					if (std::next(it) == end_) {
						break;
					}
					T tmp = std::abs(*std::next(it) - *it);
					if (tmp > max_change) {
						max_change = tmp;
					}
				}
				return max_change;
			}
		}
	}
}

#endif