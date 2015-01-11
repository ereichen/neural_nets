#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>

namespace neural_nets
{
	namespace detail
	{
		namespace random_utils
		{
			template <typename T>
			T value_in_range(T const &lower_, T const &upper_)
			{
				using distribution = std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>::type;
				distribution dist(lower_, upper_);
				static std::default_random_engine random_engine(std::random_device{}());
				return dist(random_engine);
			}

			template <typename T>
			bool true_with_probability(T const &chance)
			{
				size_t val = static_cast<size_t>(100 - 100 * chance);
				return value_in_range<size_t>(0, 100) >= val;
			}

			template <typename T>
			T normal_distributed_value(T const &mean, T const &variance)
			{
				static std::random_device rd;
				static std::mt19937 gen(rd());
				std::normal_distribution<T> d(mean, std::sqrt(variance));
				return d(gen);
			}

		}
	}
}



#endif