#ifndef NEURAL_EXCEPTION_H
#define NEURAL_EXCEPTION_H

#include <exception>
#include <string>

namespace neural_nets
{
	class neural_exception : public std::exception
	{
	public:
		explicit neural_exception(std::string const &message_) : message(message_) {}
		virtual ~neural_exception() {}

		virtual const char* what() const throw() { return message.c_str(); }

	private:
		std::string message;
	};
}

#endif