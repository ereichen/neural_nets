#ifndef NET_TRAINING_H
#define NET_TRAINING_H

#include <algorithm>

#include "neural_nets\general_net.h"
#include "neural_nets\detail\net_initialization.h"
#include "neural_nets\detail\jacobian_calculation.h"
#include "neural_nets\training_options.h"


namespace neural_nets
{
	template <typename T, typename dynamic_system>
	T train_lm(dynamic_system sys_, boost::numeric::ublas::matrix<T> const &inputs_, boost::numeric::ublas::matrix<T> const &outputs_, std::vector<T> &best_weights_)
	{
		return train_lm(sys_, inputs_, outputs_, best_weights_, lm_options<T>());
	}

	template <typename T, typename dynamic_system>
	T train_lm(dynamic_system sys_, boost::numeric::ublas::matrix<T> const &inputs_, boost::numeric::ublas::matrix<T> const &desired_outputs_, std::vector<T> &best_weights_, lm_options<T> const &opts_)
	{
		using namespace boost::numeric::ublas;
		T lambda = 1.0;

		std::vector<T> paras(sys_.get_parameter_count());
		sys_.get_parameters(paras.begin(), paras.end());
		std::vector<T> best_paras = paras;

		size_t iterations = 0, n = inputs_.size1()*sys_.get_output_count();
		bool new_weights = true;
		matrix<T> jacobian, left_side, hessian_approx;
		vector<T> solution_vector(n), output(desired_outputs_.size2());
		T min_error = std::numeric_limits<T>::max(), current_error;
		std::deque<T> error_history(opts_.rel_tol_horizont, min_error/opts_.rel_tol_horizont);

		while (true) {
			sys_.set_parameters(paras.begin(), paras.end());

			if (new_weights) {

				jacobian = detail::calc_jacobian_numerically(sys_, inputs_, opts_);
				hessian_approx = prod(trans(jacobian), jacobian);
				left_side = hessian_approx;

				current_error = 0;
				solution_vector = boost::numeric::ublas::vector<T>(n);
				size_t cnt = 0;
				for (size_t i = 0; i < inputs_.size1(); ++i) {

					sys_(std::next(inputs_.begin1(), i).begin(), std::next(inputs_.begin1(), i).end(), 
						output.begin(), output.end());
					for (size_t j = 0; j < output.size(); ++j) {
						solution_vector(cnt) = desired_outputs_(i, j) - output(j);
						current_error += solution_vector(cnt)*solution_vector(cnt);
						++cnt;
					}
				}
				current_error /= inputs_.size1();
				if (std::isnan(current_error) || std::isinf(current_error)) {
					current_error = std::numeric_limits<T>::max();
				}
				if (!iterations) {
					min_error = current_error;
				}
				sys_.clear_internal_memory();
				solution_vector = prod(trans(jacobian), solution_vector);
			}

			for (size_t i = 0; i < left_side.size1(); ++i) {
				left_side(i, i) = hessian_approx(i, i) + lambda*hessian_approx(i, i);
			}
			error_history.pop_front();
			error_history.push_back(current_error);
			T error_change = detail::math_utils::maximum_change(error_history.begin(), error_history.end());

			if (opts_.display_iterations) {
				std::cout << iterations << '\t' << current_error << "\t\t" << lambda << "\t\t" << error_change << '\n';
			}

			if (current_error < opts_.abs_tol || iterations >= opts_.max_iterations || error_change < opts_.rel_tol) {
				break;
			}

			auto delta = detail::matrix_utils::solve_linear_equation_system(left_side, solution_vector);

			std::vector<T> new_paras;
			new_paras.reserve(paras.size());
			for (size_t i = 0; i < paras.size(); ++i) {
				new_paras.push_back(paras[i] + delta(i));
			}

			T new_error = 0;
			sys_.set_parameters(new_paras.begin(), new_paras.end());
			for (size_t i = 0; i < inputs_.size1(); ++i) {
				sys_(std::next(inputs_.begin1(), i).begin(), std::next(inputs_.begin1(), i).end(), 
					output.begin(), output.end());
				T tmp = 0;
				for (size_t j = 0; j < output.size(); ++j) {
					tmp += (desired_outputs_(i, j) - output(j))*(desired_outputs_(i, j) - output(j));
				}
				new_error += tmp;
			}
			new_error /= inputs_.size1();
			if (std::isnan(new_error) || std::isinf(new_error)) {
				new_error = std::numeric_limits<T>::max();
			}
			sys_.clear_internal_memory();

			if (!std::isnan(new_error) && new_error < current_error) {
				lambda /= opts_.lambda_dec_factor;
				paras = new_paras;
				if (new_error < min_error) {
					best_paras = paras;
					min_error = new_error;
				}
				new_weights = true;
			}
			else {
				if (lambda <= opts_.max_lambda)
					lambda *= opts_.lambda_inc_factor;
				new_weights = false;
			}

			++iterations;
		}
		sys_.clear_internal_memory();
		sys_.set_parameters(best_paras.begin(), best_paras.end());
		best_weights_ = best_paras;
		return min_error;
	}


	template <typename dynamic_system, typename T>
	T train_lm_stepwise(dynamic_system &sys_, boost::numeric::ublas::matrix<T> const &u_, 
		boost::numeric::ublas::matrix<T> const &y_, 
		boost::numeric::ublas::matrix<T> const &u_valid_, 
		boost::numeric::ublas::matrix<T> const &y_valid_, 
		lm_step_options<T> const &step_opts_ = lm_step_options<T>())
	{
		using namespace boost::numeric::ublas;

		size_t step_size = std::min(u_.size1(), static_cast<size_t>(std::abs(step_opts_.step_percentage)*static_cast<T>(u_.size1())));

		detail::output_neuron_initializer<dynamic_system, T> output_initializer(sys_, y_);

		std::vector<T> weights, best_weights, tmp_best_weights;
		size_t longest_trial = 0, best_trial = 0;
		T err_total_best = std::numeric_limits<T>::max();

		for (size_t i = 1; i <= step_opts_.max_iterations; ++i) {

			// Todo: make seperate function in detail
			if (step_opts_.init_weights_random) {
				T err_weight_init = std::numeric_limits<T>::max();
				std::vector<T> best_init_weights(sys_.get_parameter_count());
				for (size_t j = 0; j < step_opts_.random_samples_per_iteration; ++j) {
					sys_.init_random(step_opts_.min_random, step_opts_.max_random);

					if (step_opts_.init_output_weights_special) {
						output_initializer.perform_init_on(sys_);
					}
					T err_weight_init_cur = detail::calculate_weight_error(sys_, u_, y_);
					sys_.clear_internal_memory();
					if (err_weight_init_cur < err_weight_init) {
						err_weight_init = err_weight_init_cur;
						sys_.get_parameters(best_init_weights.begin(), best_init_weights.end());
					}
				}
				sys_.set_parameters(best_init_weights.begin(), best_init_weights.end());
			}
			// end todo

			T err_best = std::numeric_limits<T>::max(), err_valid = std::numeric_limits<T>::max();

			size_t j;
			for (j = std::min(step_size, u_.size1()); j <= u_.size1(); j = std::min(u_.size1(), j + step_size)) {
				matrix<T> u_tmp(j, u_.size2()), y_tmp(j, y_.size2());
				for (size_t k = 0; k < j; ++k) {
					for (size_t h = 0; h < u_.size2(); ++h) {
						u_tmp(k, h) = u_(k, h);
					}
					for (size_t h = 0; h < y_.size2(); ++h) {
						y_tmp(k, h) = y_(k, h);
					}
				}
				T err_cur = train_lm(sys_, u_tmp, y_tmp, weights, step_opts_.lm_opts);

				sys_.clear_internal_memory();
				sys_.set_parameters(weights.begin(), weights.end());
				if (err_cur < err_best && j == u_.size1()) {
					err_best = err_cur;
					tmp_best_weights = weights;
					break;
				}
				if (err_cur > std::numeric_limits<T>::max()/100) {
					break;
				}
			}
			T err_train = err_best;
			if (u_valid_.size1() > 0) {
				dynamic_system sys_tmp(sys_);
				err_valid = detail::math_utils::normalized_error(sys_tmp(u_valid_), y_valid_);
				err_best += err_valid;
			}

			if (err_best < err_total_best && j >= longest_trial) {

				err_total_best = err_best;
				longest_trial = j;
				best_trial = i;
				best_weights = tmp_best_weights;

				if (step_opts_.display_iterations) {
					std::cout << "\rTrial Nr. " << i << ", Training Error: " << err_train;
					if (u_valid_.size1() > 0) {
						std::cout << ", Validation Error: " << err_valid;
					}
					std::cout << '\n';
				}
				if (err_total_best < step_opts_.abs_tol) {
					break;
				}
			}
			else {
				if (step_opts_.display_iterations) {
					std::cout << "\r" << i << " of " << step_opts_.max_iterations << " Trials";
				}
			}

		}
		if (best_weights.empty()) {
			best_weights = weights;
		}
		sys_.set_parameters(best_weights.begin(), best_weights.end());
		if (step_opts_.display_iterations) {
			std::cout << "\nBest Trial: " << best_trial << " with total error: " << err_total_best << " after " << longest_trial << " of " << u_.size1() << " samples\n";
		}
		return err_total_best;
	}

	template <typename dynamic_system, typename T>
	T train_lm_stepwise(dynamic_system &sys_, 
		boost::numeric::ublas::matrix<T> const &u_, 
		boost::numeric::ublas::matrix<T> const &y_, 
		lm_step_options<T> const &step_opts_ = lm_step_options<T>())
	{
		return train_lm_stepwise(sys_, u_, y_, boost::numeric::ublas::matrix<T>(), boost::numeric::ublas::matrix<T>(), step_opts_);
	}
}


#endif