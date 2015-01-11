#include <iostream> // For output

#include "neural_nets\general_net.h" // General Dynamic Neural Network (GDNN) class template
#include "neural_nets\net_training.h" // Neural Network training methods (Levenberg-Marquardt)
#include "neural_nets\net_signals.h" // Optimal APRBS (training signal) generation

int main()
{
	using namespace neural_nets; // Neural network library
	using namespace boost::numeric::ublas; // Boost vector and matrix libraries

	// Create a neural network with 5 neurons (we need 2 inputs, 2 hidden, 1 output neurons)
	general_net<double> net(5);

	// Connect these neurons
	net.connect_neurons(0, 2);
	net.connect_neurons(0, 3);
	net.connect_neurons(1, 2);
	net.connect_neurons(1, 3);
	net.connect_neurons(2, 4);
	net.connect_neurons(3, 4);

	// Declare input and output neurons
	net.declare_as_input(0);
	net.declare_as_input(1);
	net.declare_as_output(4);

	// XOR truth table
	matrix<double> x(4, 2);
	x(0, 0) = 0; x(0, 1) = 0; // 0 0
	x(1, 0) = 0; x(1, 1) = 1; // 0 1
	x(2, 0) = 1; x(2, 1) = 0; // 1 0
	x(3, 0) = 1; x(3, 1) = 1; // 1 1
	matrix<double> y(4, 1);
	y(0, 0) = 1; // 1
	y(1, 0) = 0; // 0
	y(2, 0) = 0; // 0
	y(3, 0) = 1; // 1

	// Perform training to make the net produce output y based on input x
	auto total_error = train_lm_stepwise(net, x, y);

	// Calculate that output of the trained net
	auto y_ident = net(x);

	// Output the results
	std::cout << "\nA\tB\tY = A xor B\tNetwork Output\n";
	std::cout << "----------------------------------------------\n";
	for (size_t i = 0; i < x.size1(); ++i) {
		std::cout << x(i, 0) << '\t' << x(i, 1) << '\t' << y(i, 0) << "\t\t" << std::round(std::abs(y_ident(i, 0))) << '\n';
	}
	std::cout << "\nTotal RMSD (Model Error): " << total_error << '\n';
}