#ifndef NEURON_H
#define NEURON_H

#include <deque>

namespace neural_nets
{
	template <class T>
	class neuron
	{
	public:
		explicit neuron(size_t index_) : index(index_), input(false), output(false) {}

		size_t get_index() const { return index; }
		size_t get_memory_size() const { return memory.size(); }

		bool is_input() const { return input; }
		bool is_output() const { return output; }
		bool has_memory() const { return !memory.empty(); }

		void set_as_input(bool input_) { input = input_; }
		void set_as_output(bool output_) { output = output_; }
		void set_memory_size(size_t size_) { memory.resize(size_); clear_internal_memory(); }
		void clear_internal_memory() { std::fill(memory.begin(), memory.end(), T(0)); }
		void add_to_memory(T const &value_) { memory.pop_back();  memory.push_front(value_); }

		T read_from_memory(size_t time_step_) const { return memory[time_step_]; }
		T output_function(T const &input_) const { return input || output ? input_ : std::tanh(input_); }

	private:
		T logistic_function(T const &input_) const { return T(1)/(T(1) + std::exp(-input_)); }
		size_t index;
		bool input, output;
		std::deque<T> memory;
	};
}

#endif