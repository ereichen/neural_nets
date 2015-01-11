#ifndef JACOBIAN_CALCULATION_H
#define JACOBIAN_CALCULATION_H

#include "neural_nets\training_options.h"

namespace neural_nets
{
	namespace detail
	{
		template <typename T, typename sys_type>
		boost::numeric::ublas::matrix<T> calc_jacobian_numerically(sys_type const &sys_, boost::numeric::ublas::matrix<T> const &inputs_, lm_options<T> const &options_)
		{

			size_t out_cnt = sys_.get_output_count();
			std::vector<T> weights(sys_.get_parameter_count());
			sys_.get_parameters(weights.begin(), weights.end());
			boost::numeric::ublas::matrix<T> jacobian(inputs_.size1()*sys_.get_output_count(), sys_.get_parameter_count());

			auto jacobian_for_body = [&](size_t i) {
				sys_type sys(sys_);
				sys_type tmp_sys(sys_);

				std::vector<T> tmp_weights = weights, out_before(out_cnt), out_after(out_cnt);
				T epsilon = detail::math_utils::calc_optimal_epsilon(tmp_weights[i]);
				tmp_weights[i] -= epsilon;
				tmp_sys.set_parameters(tmp_weights.begin(), tmp_weights.end());

				for (size_t j = 0; j < inputs_.size1(); ++j) {

					sys(std::next(inputs_.begin1(), j).begin(), std::next(inputs_.begin1(), j).end(),
						out_before.begin(), out_before.end());
					tmp_sys(std::next(inputs_.begin1(), j).begin(), std::next(inputs_.begin1(), j).end(), out_after.begin(), out_after.end());

					for (size_t k = 0; k < out_cnt; ++k) {
						jacobian(j*out_cnt + k, i) = (out_before[k] - out_after[k]) / epsilon;
					}
				}
			};

			if (options_.use_parallelization) {
#pragma omp parallel for
				for (size_t i = 0; i < jacobian.size2(); ++i) { jacobian_for_body(i); }
			}
			else {
				for (size_t i = 0; i < jacobian.size2(); ++i) { jacobian_for_body(i); }
			}
			return jacobian;
		}
	}
}

#endif