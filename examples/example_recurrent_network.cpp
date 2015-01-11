#include <iostream> // For output
#include <iomanip> // For output formatting
#include "neural_nets\neural_nets.h" // All relevant headers for full neural network usage

int main()
{
	using namespace neural_nets; // Neural network library
	using namespace boost::numeric::ublas; // Boost vector and matrix libraries

	// Create a neural network with 4 neurons
	general_net<double> net(4);

	// Connect these neurons
	net.connect_neurons(0, 1);
	net.connect_neurons(0, 2);
	net.connect_neurons(1, 3);
	net.connect_neurons(2, 3);

	// Create a delay line with 1 delay unit
	tapped_delay_line<double> tdl(1);

	// Create recurrent connections in the network
	net.connect_neurons(1, 0, tdl);
	net.connect_neurons(2, 0, tdl);

	// Declare input and output neurons
	net.declare_as_input(0);
	net.declare_as_output(3);

	// Create a time vector from t = 0.0 to 200.0 with 200 steps
	auto t = net_signals::linspace(0.0, 200.0, 200);

	// Create an optimal APRPS excitation signal with length of t, maximum hold time of 20.0 and interval between -1.0 to 1.0
	auto u = net_signals::amp_pseudo_random_binary_sequence(t, 20.0, -1.0, 1.0);

	// Create a training output signal (here a first order low pass filter with gain = 1.0 and time constant = 3.0)
	auto y = net_signals::low_pass_filter(t, u, 1.0, 3.0);

	// Levenberg-Marquardt optimization options
	lm_options<double> lm_opts;
	lm_opts.abs_tol = 1.0e-6; // Absolute tolerance
	lm_opts.rel_tol = 1.0e-6; // Relative tolerance
	lm_opts.rel_tol_horizont = 15; // Horizon for computing the relative tolerance change
	lm_opts.display_iterations = false; // Don't show optimization steps

	// Multi-trial training options
	lm_step_options<double> step_opts;
	step_opts.abs_tol = 1.0e-5; // Absolute tolerance for trials
	step_opts.lm_opts = lm_opts; // Levenberg-Marquardt training options (per trial)
	step_opts.display_iterations = true; // Show optimization trials

	// Perform training to make the net produce output y based on input u
	auto total_error = train_lm_stepwise(net, u, y, step_opts);

	// Calculate that output of the trained net
	auto y_ident = net(u);

	// Output the results
	std::cout << std::setprecision(6) << std::fixed;
	std::cout << "\n\nReal Output:\tModel Output:\tRMSD (Model Error):\n";
	for (size_t i = 0; i < y.size1(); ++i) {
		std::cout << y(i, 0) << '\t' << y_ident(i, 0) << '\t' << std::pow(y(i, 0) - y_ident(i, 0), 2) << '\n';
	}
	std::cout << "Total RMSD (Model Error): " << total_error << '\n';
}