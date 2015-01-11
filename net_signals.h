#ifndef NET_SIGNALS_H
#define NET_SIGNALS_H

#include <vector>
#include "neural_nets\detail\random_utils.h"
#include "neural_nets\detail\matrix_utils.h"
#include "neural_nets\detail\linear_feedback_shift_register.h"

namespace neural_nets
{
	namespace net_signals
	{
		template <typename T>
		boost::numeric::ublas::vector<T> linstep_space(T const &start_, T const &end_, T const &step_size_)
		{
			T tmp = start_;
			boost::numeric::ublas::vector<T> result(static_cast<size_t>((end_ - start_) / step_size_ + 1), 1);
			for (size_t i = 0; i < result.size(); ++i) {
				result(i) = tmp;
				tmp += step_size_;
			}
			return result;
		}

		template <typename T>
		boost::numeric::ublas::vector<T> linspace(T const &start_, T const &end_, size_t const lenght_)
		{
			boost::numeric::ublas::vector<T> result(lenght_, 1);
			T step_size = (end_ - start_) / (lenght_ - 1);
			T current_value = start_;
			for (size_t i = 0; i < lenght_; i++) {
				result(i) = current_value;
				current_value += step_size;
			}
			return result;
		}

		template <typename T>
		boost::numeric::ublas::matrix<T> init_with_value(size_t const length_, T const &value_, size_t dim_ = 1)
		{
			boost::numeric::ublas::matrix<T> result(length_, dim_);
			for (size_t i = 0; i < length_; ++i) {
				for (size_t j = 0; j < dim_; ++j) {
					result(i, j) = value_;
				}
			}
			return result;
		}


		template <typename T>
		boost::numeric::ublas::matrix<T> amp_pseudo_random_binary_sequence(boost::numeric::ublas::vector<T> const &t_, T max_hold_time_, T min_, T max_, size_t dim_ = 1)
		{
			boost::numeric::ublas::matrix<T> result(t_.size(), dim_);
			std::vector<T> t;
			t.reserve(t_.size());
			for (size_t i = 0; i < t_.size(); ++i) {
				t.push_back(t_(i));
			}
			std::vector<T> in = detail::amp_pseudo_random_binary_sequence(t, max_hold_time_, min_, max_);

			for (size_t i = 0; i < dim_; ++i) {
				for (size_t j = 0; j < in.size(); ++j) {
					result(j, i) = in[j];
				}
			}
			return result;
		}

		template <typename T>
		boost::numeric::ublas::matrix<T> low_pass_filter(
			boost::numeric::ublas::vector<T> const &time,
			boost::numeric::ublas::matrix<T> const &input, 
			T const &gain, 
			T const &time_constant)
		{
			boost::numeric::ublas::matrix<T> result(input.size1(), input.size2());
			for (size_t j = 0; j < input.size2(); ++j) {
				result(0, j) = 0;
			}
			for (size_t i = 1; i < input.size1(); ++i) {
				T dt = time(i) - time(i - 1);
				for (size_t j = 0; j < input.size2(); ++j) {
					result(i, j) = 1 / (time_constant / dt + 1)*(gain*input(i, j) + (time_constant / dt)*result(i - 1, j));
				}
			}

			return result;
		}
	}
}


#endif