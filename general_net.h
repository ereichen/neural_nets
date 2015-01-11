#ifndef GENERAL_NET_H
#define GENERAL_NET_H

#include <sstream>
#include <map>

#include "neural_nets\detail\matrix_utils.h"
#include "neural_nets\neuron.h"
#include "neural_nets\neural_exception.h"
#include "neural_nets\tapped_delay_line.h"
#include "neural_nets\detail\random_utils.h"
#include "neural_nets\detail\math_utils.h"

namespace neural_nets
{

	template <class T>
	class general_net
	{
	public:
		explicit general_net() {};
		explicit general_net(size_t neuron_count_);

		size_t get_neuron_count() const { return neurons.size(); }
		size_t get_input_count() const { return input_count; }
		size_t get_output_count() const { return output_count; }
		size_t get_parameter_count() const { return weight_count; }

		void declare_as_input(size_t index_);
		void declare_as_output(size_t index_);
		void set_neuron_bias_weight(size_t index_, T const &weight_) { biases[index_] = weight_; }
		void connect_neurons(size_t first_, size_t second_, T const &weight_ = 1.0);
		void connect_neurons(size_t first_, size_t second_, tapped_delay_line<T> const &tdl_);
		void set_connection_weight(size_t from_neuron_, size_t to_neuron_, size_t tdl_index_, T weight_);
		void clear_internal_memory() { for (auto &i : neurons) i.clear_internal_memory(); }
		void init_random(T const &lower_, T const &upper_);
		void init_bias_weights_random(T const &lower_, T const &upper_);

		T get_neuron_bias_weight(size_t neuron_index_) const;
		T get_connection_weight(size_t from_neuron_, size_t to_neuron_, size_t tdl_index_) const;

		neuron<T> const &get_neuron(size_t index_) const { return neurons[index_]; }
		boost::numeric::ublas::matrix<tapped_delay_line<T>> const &get_adjacency_matrix() const { return connections; }

		template<typename iter> void set_parameters(iter begin_, iter end_);
		template<typename iter> void get_parameters(iter begin_, iter end_) const;

		void operator()(T const &input_, T &output_); // SISO
		template<typename iter> void operator()(iter input_begin_, iter input_end_, T &output_); // MISO
		template<typename iter> void operator()(T const &input_, iter output_begin_, iter output_end_); // SIMO
		template<typename iter1, typename iter2> void operator()(iter1 input_begin_, iter1 input_end_, iter2 output_begin_, iter2 output_end_); //MIMO
		boost::numeric::ublas::matrix<T> operator()(boost::numeric::ublas::matrix<T> const &u_);

		bool has_unused_neurons() const;
		bool is_valid() const;

	private:
		bool sort_required;
		size_t input_count, output_count, weight_count;
		std::map<size_t, size_t> input_order;
		std::vector<T> biases;
		std::vector<size_t> sorted_indices;
		std::vector<neuron<T>> neurons;
		boost::numeric::ublas::matrix<tapped_delay_line<T>> connections;

		bool contains_element(std::vector<size_t> const &vec_, size_t const &value_) const;
		size_t find_missing_entry(std::vector<size_t> vec_) const; // Yes, call by value
		size_t parse_line(size_t line_, std::vector<size_t> &stack_) const;
		std::string get_algebraic_loop_string(std::vector<size_t> const &stack_, size_t to_) const;
		void topological_sort();
	};




	template<class T>
	general_net<T>::general_net(size_t neuron_count_) : connections(neuron_count_, neuron_count_), sort_required(true), weight_count(neuron_count_), input_count(0), output_count(0)
	{
		neurons.reserve(neuron_count_);
		biases.reserve(neuron_count_);
		sorted_indices.reserve(neuron_count_);
		for (size_t i = 0; i < neuron_count_; ++i) {
			biases.push_back(1);
			neurons.emplace_back(i);
			sorted_indices.push_back(i);
		}
	}

	template<class T>
	void general_net<T>::declare_as_input(size_t index_)
	{
		if (!neurons[index_].is_input()) {
			sort_required = true;
			input_order[index_] = input_count;
			++input_count;
			neurons[index_].set_as_input(true);
		}
	}

	template<class T>
	void general_net<T>::declare_as_output(size_t index_)
	{
		if (!neurons[index_].is_output()) {
			sort_required = true;
			++output_count;
			neurons[index_].set_as_output(true);
		}
	}

	template <class T>
	void general_net<T>::init_random(T const&lower_, T const &upper_)
	{
		std::vector<T> weights(weight_count);
		for (auto &i : weights) {
			i = detail::random_utils::value_in_range<T>(lower_, upper_);
		}
		set_parameters(weights.begin(), weights.end());
	}

	template <class T>
	void general_net<T>::init_bias_weights_random(T const&lower_, T const &upper_)
	{
		for (auto &i : biases) {
			i = detail::random_utils::value_in_range<T>(lower_, upper_);
		}
	}

	template<class T>
	void general_net<T>::connect_neurons(size_t first_, size_t second_, T const &weight_ = 1.0)
	{
		connect_neurons(first_, second_, tapped_delay_line<T>(0, weight_));
	}

	template<class T>
	void general_net<T>::connect_neurons(size_t first_, size_t second_, tapped_delay_line<T> const &tdl_)
	{
		if (tdl_.has_delays()) {
			if (tdl_.get_maximum_delay() + 1 > neurons[first_].get_memory_size()) {
				neurons[first_].set_memory_size(tdl_.get_maximum_delay() + 1);
			}
		}
		sort_required = true;
		connections(second_, first_) = tdl_;
		weight_count += tdl_.get_delay_count();
	}

	template<class T>
	T general_net<T>::get_neuron_bias_weight(size_t neuron_index_) const
	{
		return biases[neuron_index_];
	}

	template<class T>
	void general_net<T>::set_connection_weight(size_t from_neuron_, size_t to_neuron_, size_t tdl_index_, T weight_)
	{
		connections(from_neuron_, to_neuron_).set_delay_by_index(tdl_index_, weight_);
	}

	template<class T>
	T general_net<T>::get_connection_weight(size_t from_neuron_, size_t to_neuron_, size_t tdl_index_) const
	{
		return connections(from_neuron_, to_neuron_).get_delay_weight(tdl_index_);
	}

	template<class T>
	template<typename iter> void general_net<T>::set_parameters(iter begin_, iter end_)
	{
		size_t neuron_count = get_neuron_count();
		for (size_t i = 0; i < neuron_count; ++i) {
			for (size_t j = 0; j < neuron_count; ++j) {
				if (connections(j, i).is_connected()) {
					for (size_t k = 0; k < connections(j, i).get_delay_count(); k++) {
						set_connection_weight(j, i, k, *begin_);
						++begin_;
					}
				}
			}
		}
		for (size_t i = 0; i < neuron_count; i++) {
			biases[i] = *begin_;
			++begin_;
		}
	}

	template<class T>
	template<typename iter> void general_net<T>::get_parameters(iter begin_, iter end_) const
	{
		size_t neuron_count = get_neuron_count();
		for (size_t i = 0; i < neuron_count; ++i) {
			for (size_t j = 0; j < neuron_count; ++j) {
				if (connections(j, i).is_connected()) {
					for (size_t k = 0; k < connections(j, i).get_delay_count(); ++k) {
						*begin_ = get_connection_weight(j, i, k);
						++begin_;
					}
				}
			}
		}
		for (size_t i = 0; i < neuron_count; i++) {
			*begin_ = get_neuron_bias_weight(i);
			++begin_;
		}
	}

	template<class T>
	boost::numeric::ublas::matrix<T> general_net<T>::operator()(boost::numeric::ublas::matrix<T> const &u_)
	{
		boost::numeric::ublas::matrix<T> y(u_.size1(), output_count);
		for (size_t i = 0; i < u_.size1(); ++i) {
			(*this)(std::next(u_.begin1(), i).begin(), std::next(u_.begin1(), i).end(), 
				std::next(y.begin1(), i).begin(), std::next(y.begin1(), i).end());
		}
		return y;
	}

	template<class T>
	void general_net<T>::operator()(T const &input_, T &output_)
	{
		auto in = std::vector<T>{input_}, out = std::vector<T>{output_};
		(*this)(in.begin(), in.end(), out.begin(), out.end());
		output_ = out.front();
	}

	template<class T>
	template<typename iter> void general_net<T>::operator()(iter input_begin_, iter input_end_, T &output_)
	{
		auto out = std::vector<T>{output_};
		(*this)(input_begin_, input_end_, out.begin(), out.end());
		output_ = out.front();
	}

	template<class T>
	template<typename iter> void general_net<T>::operator()(T const &input_, iter output_begin_, iter output_end_)
	{
		auto in = std::vector<T>{input_};
		(*this)(in.begin(), in.end(), output_begin_, output_end_);
	}

	template<class T>
	template<typename iter1, typename iter2> void general_net<T>::operator()(iter1 input_begin_, iter1 input_end_, iter2 output_begin_, iter2 output_end_)
	{
		topological_sort();
		std::vector<T> output;
		size_t neuron_count = get_neuron_count();
		output.resize(neuron_count);

		for (size_t k = 0; k < neuron_count; ++k) {
			size_t i = sorted_indices[k];
			bool input_detected = false;
			for (size_t j = 0; j < neuron_count; ++j) {
				if (neurons[i].is_input() && !input_detected) {
					output[i] = *std::next(input_begin_, input_order.find(i)->second);
					input_detected = true;
				}
				if (connections(i, j).is_connected()) {
					if (connections(i, j).is_instant())
						output[i] += connections(i, j).get_delay_weight(0)*output[j];
					if (connections(i, j).has_delays()) {
						size_t delay_weight_counter = 0;
						for (auto const &h : connections(i, j).get_delay_line()) {
							if (h.delay_index) {
								output[i] += connections(i, j).get_delay_weight(delay_weight_counter)*neurons[j].read_from_memory(h.delay_index - 1);
							}
							++delay_weight_counter;
						}
					}
				}
			}
			output[i] = neurons[i].output_function(output[i] + biases[i]);
		}

		for (size_t i = 0; i < neuron_count; ++i) {
			if (neurons[i].has_memory())
				neurons[i].add_to_memory(output[i]);
			if (neurons[i].is_output()) {
				*output_begin_ = output[i];
				++output_begin_;
			}
		}
	}

	template <class T>
	size_t general_net<T>::find_missing_entry(std::vector<size_t> vec_) const
	{
		std::sort(vec_.begin(), vec_.end());
		size_t next = 0;
		for (auto it = vec_.begin(); it != vec_.end(); ++it) {
			if (*it != next) {
				return next;
			}
			++next;
		}
		return next;
	}

	template <class T>
	bool general_net<T>::contains_element(std::vector<size_t> const &vec_, size_t const &value_) const
	{
		return std::find(vec_.begin(), vec_.end(), value_) != vec_.end();
	}

	template <class T>
	std::string general_net<T>::get_algebraic_loop_string(std::vector<size_t> const &stack_, size_t to_) const
	{
		std::stringstream ss;
		ss << "Algebraic loop detected: " << stack_.front() << " -> ";
		if (stack_.size() > 1) {
			ss << to_ << " -> ";
			for (auto i = stack_.rbegin(), e = std::next(stack_.rend(), -1); i != e; ++i) {
				ss << *i << " -> ";
			}
		}
		ss << stack_.front() << " -> " << "infinite loop!";
		return ss.str();
	}

	template <class T>
	size_t general_net<T>::parse_line(size_t line_, std::vector<size_t> &stack_) const
	{
		for (size_t i = 0; i < get_neuron_count(); i++) {
			if (connections(line_, i).is_instant()) {
				if (!contains_element(sorted_indices, i)) {
					if (contains_element(stack_, i)) {
						throw neural_exception(get_algebraic_loop_string(stack_, line_));
					}
					stack_.push_back(line_);
					return parse_line(i, stack_);
				}
			}
		}
		return line_;
	}

	template<class T>
	bool general_net<T>::has_unused_neurons() const
	{
		for (size_t col = 0; col < get_neuron_count(); ++col) {
			bool has_outputs = false;
			for (size_t row = 0; row < get_neuron_count(); ++row) {
				if (connections(row, col).is_connected()) {
					has_outputs = true;
					break;
				}
			}
			if (!has_outputs && !neurons[col].is_output()) {
				return true;
			}
		}
		for (size_t row = 0; row < get_neuron_count(); ++row) {
			bool has_inputs = false;
			for (size_t col = 0; col < get_neuron_count(); ++col) {
				if (connections(row, col).is_connected()) {
					has_inputs = true;
					break;
				}
			}
			if (!has_inputs && !neurons[row].is_input()) {
				return true;
			}
		}
		return false;
	}

	template<class T>
	bool general_net<T>::is_valid() const
	{
		auto tmp(*this);
		try {
			tmp.topological_sort();
		}
		catch (neural_exception const &) {
			return false;
		}
		return true;
	}

	template <class T>
	void general_net<T>::topological_sort()
	{
		if (!sort_required) {
			return;
		}
		bool input_detected = false, output_detected = false;
		for (auto const &i : neurons) {
			if (i.is_input()) {
				input_detected = true;
			}
			if (i.is_output()) {
				output_detected = true;
			}
		}
		if (!input_detected) {
			throw neural_exception("Network has no inputs!");
		}
		if (!output_detected) {
			throw neural_exception("Network has no outputs!");
		}
		if (has_unused_neurons()) {
			throw neural_exception("Network contains neurons which have neither an input nor an output!");
		}

		sorted_indices.clear();
		size_t current_line = 0;
		std::vector<size_t> stack;
		while (sorted_indices.size() < get_neuron_count()) {
			sorted_indices.push_back(parse_line(current_line, stack));
			if (stack.empty()) {
				current_line = find_missing_entry(sorted_indices);
			}
			else {
				current_line = stack.front();
				stack.clear();
			}
		}
		sort_required = false;
	}


	template <typename T>
	std::ostream &operator<<(std::ostream &stream, neural_nets::general_net<T> const &net)
	{
		std::vector<size_t> inputs, outputs;
		inputs.reserve(net.get_input_count());
		outputs.reserve(net.get_output_count());
		for (size_t i = 0; i < net.get_neuron_count(); ++i) {
			if (net.get_neuron(i).is_input()) {
				inputs.push_back(i);
			}
			if (net.get_neuron(i).is_output()) {
				outputs.push_back(i);
			}
		}

		net.get_input_count() > 1 ? stream << net.get_input_count() << " Input Neurons: " : stream << net.get_input_count() << " Input Neuron: ";
		for (size_t i = 0; i < inputs.size(); ++i) {
			i < inputs.size() - 1 ? stream << inputs[i] << ", " : stream << inputs[i];
		}
		stream << '\n';
		net.get_output_count() > 1 ? stream << net.get_output_count() << " Output Neurons: " : stream << net.get_output_count() << " Output Neuron: ";
		for (size_t i = 0; i < outputs.size(); ++i) {
			i < outputs.size() - 1 ? stream << outputs[i] << ", " : stream << outputs[i];
		}
		stream << "\nTotal Neurons: " << net.get_neuron_count() << '\n';
		stream << "Parameters: " << net.get_parameter_count() << '\n';
		for (size_t i = 0; i < net.get_neuron_count(); ++i) {
			stream << "Bias " << i << ": " << net.get_neuron_bias_weight(i) << '\n';
		}

		size_t max_delay = 0;
		for (size_t i = 0; i < net.get_neuron_count(); ++i) {
			for (size_t j = 0; j < net.get_neuron_count(); ++j) {
				max_delay = std::max(max_delay, net.get_adjacency_matrix()(i, j).get_maximum_delay());
			}
		}

		for (size_t i = 0; i < net.get_neuron_count(); ++i) {
			for (size_t j = 0; j < net.get_neuron_count(); ++j) {
				if (net.get_adjacency_matrix()(i, j).is_connected()) {
					for (size_t k = 0; k < net.get_adjacency_matrix()(i, j).get_delay_count(); ++k) {
						size_t delay = net.get_adjacency_matrix()(i, j).get_delay_line()[k].delay_index;
						T weight = net.get_adjacency_matrix()(i, j).get_delay_line()[k].delay_weight;
						stream << "Weight from " << j << " to " << i << " (" << delay << " delay): " << weight << '\n';
					}
				}
			}
		}

		return stream;
	}

}

#endif

























