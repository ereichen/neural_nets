#ifndef NET_INITIALIZATION_H
#define NET_INITIALIZATION_H

#include "neural_nets\general_net.h"

namespace neural_nets
{
	namespace detail
	{
		template <typename dynamic_system, typename T>
		T calculate_weight_error(dynamic_system &sys_, boost::numeric::ublas::matrix<T> const &u_, boost::numeric::ublas::matrix<T> const &y_desired_)
		{
			auto y = sys_(u_);

			T err = 0;
			for (size_t i = 0; i < y_desired_.size1(); ++i) {
				for (size_t j = 0; j < y_desired_.size2(); ++j) {
					T err_cur = y(i, j) - y_desired_(i, j);
					err += err_cur*err_cur;
				}
			}
			return err / u_.size1();
		}

		template <typename dynamic_system, typename T>
		class net_initializer {
			struct connection_info
			{
				explicit connection_info(size_t source_neuron_, size_t source_delay_) :
				source_neuron(source_neuron_), source_delay(source_delay_) {}
				size_t source_neuron, source_delay;
			};

			struct neuron_input_info
			{
				explicit neuron_input_info(size_t index_) :
				index(index_) {}
				size_t index;
				std::vector<connection_info> connection_source;
			};

		public:
			explicit net_initializer(dynamic_system const &net_, boost::numeric::ublas::matrix<T> const &y_) {
				outputs_range.reserve(y_.size2());
				neuron_inputs.reserve(y_.size2());

				for (size_t j = 0; j < y_.size2(); ++j) {
					T max_tmp = y_(0, 0), min_tmp = y_(0, 0);
					for (size_t i = 0; i < y_.size1(); ++i) {
						if (y_(i, j) > max_tmp) {
							max_tmp = y_(i, j);
						}
						if (y_(i, j) < min_tmp) {
							min_tmp = y_(i, j);
						}
					}
					outputs_range.push_back(max_tmp - min_tmp);
				}
				for (size_t i = 0; i < net_.get_neuron_count(); ++i) {
					if (net_.get_neuron(i).is_output()) {
						neuron_input_info input_info(i);
						for (size_t j = 0; j < net_.get_neuron_count(); ++j) {
							if (net_.get_adjacency_matrix()(i, j).get_delay_count()) {
								for (size_t k = 0; k < net_.get_adjacency_matrix()(i, j).get_delay_line().size(); ++k) {
									input_info.connection_source.push_back(connection_info(j, k));
								}
							}
						}
						neuron_inputs.push_back(input_info);
					}
				}
			}

			void perform_init_on(dynamic_system &net_) {

				for (size_t i = 0; i < neuron_inputs.size(); ++i) {
					size_t relevant_weight_count = detail::random_utils::value_in_range<size_t>(1, neuron_inputs[i].connection_source.size() + 1);
					size_t cnt = relevant_weight_count;

					T init_weight = outputs_range[i]/static_cast<T>(relevant_weight_count);
					for (size_t j = 0; j < neuron_inputs[i].connection_source.size(); ++j, --cnt) {
						if (!cnt) {
							break;
						}
						size_t target = neuron_inputs[i].index;
						size_t source = neuron_inputs[i].connection_source[j].source_neuron;
						size_t delay = neuron_inputs[i].connection_source[j].source_delay;
						net_.set_connection_weight(target, source, delay, init_weight);
					}
					if (cnt) {
						net_.set_neuron_bias_weight(neuron_inputs[i].index, init_weight);
					}
				}
			}

		private:
			std::vector<T> outputs_range;
			std::vector<neuron_input_info> neuron_inputs;
		};
		
		template <typename dynamic_system, typename T>
		class empty_initializer {
			public:
				explicit empty_initializer(dynamic_system const &sys_, boost::numeric::ublas::matrix<T> const &y_) {}
				void perform_init_on(dynamic_system &system_) {}
		};

		template <typename dynamic_system, typename T>
		class output_neuron_initializer : public std::conditional<std::is_same<dynamic_system, general_net<T>>::value, net_initializer<dynamic_system, T>, empty_initializer<dynamic_system, T>>::type
		{
			using parent_type = typename std::conditional<std::is_same<dynamic_system, general_net<T>>::value, net_initializer<dynamic_system, T>, empty_initializer<dynamic_system, T>>::type;

			public:
				explicit output_neuron_initializer(dynamic_system const &sys_, boost::numeric::ublas::matrix<T> const &y_) : parent_type(sys_, y_) {}
		};
	}
}

#endif