#ifndef LINEAR_FEEDBACK_SHIFT_REGISTER_H
#define LINEAR_FEEDBACK_SHIFT_REGISTER_H

#include <vector>
#include "neural_nets\detail\matrix_utils.h"

namespace neural_nets
{
	namespace net_signals
	{
		namespace detail
		{

			template <class T>
			class linear_feedback_shift_register
			{
			public:
				explicit linear_feedback_shift_register(size_t grade_, size_t clock_period_ = 1);
				T calculate_output();
				bool sequence_done() const;
				void clear_internal_memory();

			private:
				size_t period, counter, grade, clock_period, clock_counter;
				std::vector<T> internal_state;
			};

			template <class T>
			std::vector<T> pseudo_random_binary_sequence(std::vector<T> const &t_, T max_hold_time_);

			template <class T>
			std::vector<T> amp_pseudo_random_binary_sequence(std::vector<T> const &t_, T max_hold_time_, T min_, T max_);

			template <class T>
			boost::numeric::ublas::matrix<T> amp_pseudo_random_binary_sequence(boost::numeric::ublas::vector<T> const &t_, T max_hold_time_, T min_, T max_, size_t dim_ = 1);


			typedef std::vector<std::vector<size_t>> register_taps;
			register_taps taps = { {}, { 1 }, { 2, 1 }, { 3, 2 }, { 4, 3 }, { 5, 3 }, { 6, 5 },
			{ 7, 6 }, { 8, 6, 5, 4 }, { 9, 5 }, { 10, 7 }, { 11, 9 }, { 12, 6, 4, 1 },
			{ 13, 4, 3, 1 }, { 14, 5, 3, 1 }, { 15, 14 }, { 16, 15, 13, 4 }, { 17, 14 },
			{ 18, 11 }, { 19, 6, 2, 1 }, { 20, 17 }, { 21, 19 }, { 22, 21 }, { 23, 18 },
			{ 24, 23, 22, 17 }, { 25, 22 }, { 26, 6, 2, 1 }, { 27, 5, 2, 1 }, { 28, 25 },
			{ 29, 27 }, { 30, 6, 4, 1 }, { 31, 28 }, { 32, 22, 2, 1 }, { 33, 20 }, { 33, 20 },
			{ 34, 27, 2, 1 }, { 35, 33 }, { 36, 25 }, { 37, 5, 4, 3, 2, 1 }, { 38, 6, 5, 1 },
			{ 39, 35 }, { 40, 38, 21, 19 }, { 41, 38 }, { 42, 41, 20, 19 }, { 43, 42, 38, 37 },
			{ 44, 43, 18, 17 }, { 45, 44, 42, 41 }, { 46, 45, 26, 25 }, { 47, 42 },
			{ 48, 47, 21, 20 }, { 49, 40 }, { 50, 49, 24, 23 }, { 51, 50, 36, 35 }, { 52, 49 },
			{ 53, 52, 38, 37 }, { 54, 53, 18, 17 }, { 55, 31 }, { 56, 55, 35, 34 }, { 57, 50 },
			{ 58, 39 }, { 59, 58, 38, 37 }, { 60, 59 }, { 61, 60, 46, 45 }, { 62, 61, 6, 5 },
			{ 63, 62 }, { 64, 63, 61, 60 }, { 65, 47 }, { 66, 65, 57, 56 }, { 67, 66, 58, 57 },
			{ 68, 59 }, { 69, 67, 42, 40 }, { 70, 69, 55, 54 }, { 71, 65 }, { 72, 66, 25, 19 },
			{ 73, 48 }, { 74, 73, 59, 58 }, { 75, 74, 65, 64 }, { 76, 75, 41, 40 }, { 77, 76, 41, 40 },
			{ 78, 77, 59, 58 }, { 79, 70 }, { 80, 79, 43, 42 }, { 81, 77 }, { 82, 79, 47, 44 },
			{ 83, 82, 38, 37 }, { 84, 71 }, { 85, 84, 58, 57 }, { 86, 85, 74, 73 } };

			template <typename T> T xor(T const &first_, T const &second_)
			{
				return first_ && !second_ || !first_ && second_;
			}

			size_t pow2(size_t exponent_)
			{
				size_t result = 1;
				for (size_t i = 0; i < exponent_; ++i)
					result *= 2;
				return result;
			}

			size_t  size_difference(size_t first_, size_t second_)
			{
				return first_ > second_ ? first_ - second_ : second_ - first_;
			}

			size_t div_to_nearest(size_t n, size_t d)
			{
				return (n + d / 2) / d;
			}

			size_t calculate_best_grade(size_t n_, size_t max_hold_time_)
			{
				size_t grade = 1, best_grade = 1, best_len = std::numeric_limits<size_t>::max();

				while (true) {
					size_t max_len = detail::pow2(grade) - 1;

					if (max_len > n_)
						break;
					size_t factor = div_to_nearest(max_hold_time_, grade);
					size_t current_len = max_len*factor;

					if (size_difference(current_len, n_) < best_len) {
						best_len = size_difference(current_len, n_);
						best_grade = grade;
					}
					grade++;
				}
				return best_grade;
			}

			template <typename T> bool almost_equal(T const &left_, T const &right_)
			{
				return abs(left_ - right_) < 2 * std::numeric_limits<T>::epsilon();
			}


			template <class T>
			linear_feedback_shift_register<T>::linear_feedback_shift_register(size_t grade_, size_t clock_period_) :
				grade(grade_), clock_period(clock_period_), clock_counter(0)
			{
				period = (detail::pow2(grade) - 1)*clock_period;
				counter = period;
				internal_state.resize(grade, 1);
			};

			template <class T>
			T linear_feedback_shift_register<T>::calculate_output()
			{
				if (counter < period && clock_counter == clock_period) {
					clock_counter = 0;
					std::vector<size_t> polynom = detail::taps[grade];
					T feedback_value = 0;

					for (auto const &i : polynom) {
						feedback_value = detail::xor<T>(feedback_value, internal_state[i - 1]);
					}
					std::rotate(internal_state.begin(), internal_state.begin() + grade - 1, internal_state.end());
					internal_state[0] = feedback_value;
				}
				if (!counter)
					clear_internal_memory();
				--counter;
				++clock_counter;
				return internal_state.back();
			}

			template <class T>
			bool linear_feedback_shift_register<T>::sequence_done() const
			{
				return !counter;
			}

			template <class T>
			void linear_feedback_shift_register<T>::clear_internal_memory()
			{
				clock_counter = 0;
				counter = period;
				std::fill(internal_state.begin(), internal_state.end(), T(1));
			}

			template <class T>
			std::vector<T> pseudo_random_binary_sequence(std::vector<T> const &t_, T max_hold_time_)
			{
				size_t max_hold_time = 0, n = t_.size();
				for (auto const &i : t_) {
					if (i > max_hold_time_)
						break;
					max_hold_time++;
				}

				size_t grade = detail::calculate_best_grade(n, max_hold_time);
				size_t clock_period = detail::div_to_nearest(max_hold_time, grade);

				linear_feedback_shift_register<T> lfsr(grade, clock_period);
				std::vector<T> result;
				result.reserve(n);

				while (result.size() < n) {
					result.push_back(lfsr.calculate_output());
				}
				return result;
			}

			template <class T>
			std::vector<T> amp_pseudo_random_binary_sequence(std::vector<T> const &t_, T max_hold_time_, T min_, T max_)
			{
				std::vector<T> result = pseudo_random_binary_sequence(t_, max_hold_time_);
				T current_val = result.front();
				size_t interval_counter = 1;

				for (auto const &i : result) {
					if (!detail::almost_equal(i, current_val)) {
						current_val = i;
						interval_counter++;
					}
				}
				T interval_steps = (max_ - min_) / (interval_counter - 1);
				std::vector<T> intervals;
				intervals.reserve(interval_counter);
				intervals.push_back(min_);

				for (size_t i = 0; i < interval_counter - 1; ++i) {
					intervals.push_back(intervals.back() + interval_steps);
				}
				std::random_shuffle(intervals.begin(), intervals.end());
				interval_counter = 0;
				current_val = result.front();

				for (auto &i : result) {
					if (!detail::almost_equal(i, current_val)) {
						current_val = i;
						interval_counter++;
					}
					i = intervals[interval_counter];
				}

				return result;
			}

		}
	}
}


#endif