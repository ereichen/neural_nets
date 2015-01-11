#ifndef TRAINING_OPTIONS_H
#define TRAINING_OPTIONS_H

namespace neural_nets
{
	template <typename T>
	struct lm_options
	{
		size_t max_iterations = 500;
		size_t rel_tol_horizont = 10;
		size_t max_lambda = 1000000000;
		T rel_tol = 1.0e-6;
		T abs_tol = 1.0e-6;
		T lambda_inc_factor = 2.0;
		T lambda_dec_factor = 10.0;
		bool display_iterations = true;
		bool use_parallelization = true;
	};

	template <typename T>
	struct lm_step_options
	{
		bool display_iterations = true;
		bool init_weights_random = true;
		bool init_output_weights_special = false;
		size_t max_iterations = 100;
		size_t random_samples_per_iteration = 10;
		T step_percentage = 0.5;
		T abs_tol = 1.0e-3;
		T min_random = -0.5;
		T max_random = 0.5;
		lm_options<T> lm_opts;
	};
}

#endif