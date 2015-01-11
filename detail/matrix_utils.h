#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <string>
#include <sstream>

#include <boost\numeric\ublas\vector.hpp>
#include <boost\numeric\ublas\matrix.hpp>
#include <boost\numeric\ublas\io.hpp>
#include <boost\numeric\ublas\lu.hpp>


namespace neural_nets
{
	namespace detail
	{
		namespace matrix_utils
		{
			template <typename T, typename L, typename A>
			std::string to_string(boost::numeric::ublas::matrix<T, L, A> const &matrix)
			{
				std::stringstream ss;
				for (size_t i = 0; i < matrix.size1(); ++i) {
					for (size_t j = 0; j < matrix.size2(); ++j) {
						ss << matrix(i, j) << ' ';
					}
					ss << '\n';
				}
				return ss.str();
			}

			template <typename T, typename A>
			std::string to_string(boost::numeric::ublas::vector<T, A> const &vec)
			{
				std::stringstream ss;
				for (size_t i = 0; i < vec.size(); ++i) {
					ss << vec(i) << '\n';
				}
				return ss.str();
			}

			template <typename T, typename L, typename A1, typename A2>
			void solve_linear_equation_system_inplace(boost::numeric::ublas::matrix<T, L, A1> &a_matrix, boost::numeric::ublas::vector<T, A2> &solution_vector)
			{
				boost::numeric::ublas::permutation_matrix<size_t> pm(a_matrix.size1());
				boost::numeric::ublas::lu_factorize(a_matrix, pm);
				boost::numeric::ublas::lu_substitute(a_matrix, pm, solution_vector);
			}

			template <typename T, typename L, typename A1, typename A2>
			boost::numeric::ublas::vector<T, A2> solve_linear_equation_system(boost::numeric::ublas::matrix<T, L, A1> system_, boost::numeric::ublas::vector<T, A2> solution_)
			{
				using namespace boost::numeric::ublas;

				T max_val = std::numeric_limits<T>::min();
				matrix<T> system(system_);
				vector<T> solution(solution_);
				vector<T> result(solution.size());
				T pivot(0);

				for (size_t i = 0, j = 0; i < system.size1(); ++i) {
					j = i + 1;
					size_t row = i;
					for (; j < system.size1(); ++j) {
						pivot = std::abs(system(j, i));
						if (pivot > max_val) {
							max_val = pivot;
							row = j;
						}
					}

					if (row != i) {
						for (j = 0; j < system.size2(); ++j) {
							pivot = system(i, j);
							system(i, j) = system(row, j);
							system(row, j) = pivot;
						}
						pivot = solution(i);
						solution(i) = solution(row);
						solution(row) = pivot;
					}
					for (j = i + 1; j < system.size1(); ++j) {
						pivot = system(j, i) / system(i, i);
						for (size_t k = i; k < system.size1(); ++k) {
							system(j, k) -= pivot*system(i, k);
						}
						solution(j) -= pivot*solution(i);
					}
				}
				size_t n = result.size() - 1;
				result(n) = solution(n) / system(n, n);
				size_t k = n - 1;
				while (true) {
					pivot = solution(k);
					for (size_t j = k + 1; j < system.size1(); ++j) {
						pivot -= system(k, j)*result(j);
					}
					result(k) = pivot / system(k, k);
					if (!k)
						break;
					--k;
				}
				return result;

			}
		}
	}
}

#endif