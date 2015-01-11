#ifndef TAPPED_DELAY_LINE_H
#define TAPPED_DELAY_LINE_H

#include <vector>

namespace neural_nets
{
	namespace detail
	{
		template <class T>
		struct tapped_delay
		{
			tapped_delay() {}
			tapped_delay(size_t delay_index_, T delay_weight_ = 1) : delay_index(delay_index_), delay_weight(delay_weight_) {}
			size_t delay_index;
			T delay_weight;
		};
	}

	template <class T>
	class tapped_delay_line
	{
	public:
		explicit tapped_delay_line() : connected(false), instant(false) {}
		explicit tapped_delay_line(size_t index_, T const &weight_ = 1) : tapped_delay_line(std::vector<detail::tapped_delay<T>>{detail::tapped_delay<T>(index_, weight_)}) {}
		explicit tapped_delay_line(std::vector<detail::tapped_delay<T>> const &delay_line_) : connected(true), instant(false)
		{
			delay_line = delay_line_;
			if (!delay_line.front().delay_index)
				instant = true;
		}

		bool is_connected() const { return connected; }
		bool is_instant() const { return instant && connected; }
		bool has_delays() const { return (delay_line.back().delay_index > 0 && !delay_line.empty()); }

		void set_delay_by_index(size_t time_step_, T weight_) { delay_line[time_step_].delay_weight = weight_; }

		size_t get_maximum_delay() const { return delay_line.empty() ? 0 : delay_line.back().delay_index; };
		size_t get_delay_count() const { return delay_line.size(); };
		T get_delay_weight(size_t time_step_) const { return delay_line[time_step_].delay_weight; }

		std::vector<detail::tapped_delay<T>> const &get_delay_line() const { return delay_line; }

	private:
		bool connected, instant;
		std::vector<detail::tapped_delay<T>> delay_line;
	};

	template <typename T>
	std::ostream &operator<<(std::ostream &stream, tapped_delay_line<T> const &tdl)
	{
		stream << "[";
		for (size_t i = 0; i < tdl.get_delay_count(); ++i) {
			if (i < tdl.get_delay_count() - 1)
				stream << "(" << tdl.get_delay_line()[i].delay_index << ", " << tdl.get_delay_line()[i].delay_weight << "), ";
			else
				stream << "(" << tdl.get_delay_line()[i].delay_index << ", " << tdl.get_delay_line()[i].delay_weight << ")";
		}
		stream << "]";
		return stream;
	}
}

#endif