--------------------------------------------------------
Recurrent Neural Network library for C++ 
--------------------------------------------------------
Version:               1.0.0.0
Author:                Elias Reichensdoerfer
Date:                  10.01.2015

--------------------------------------------------------
About this library
--------------------------------------------------------
This library provides means to use arbitrary recurrent neural networks in C++. The recurrent neural network
is implemented as a class template and allows arbirary connections containing arbirary delay lines. For network
training, the Levenberg-Marquardt algorithm is available. Furthermore there are excitation signal generators
implemented in this library that allow convenient generation of such signals for network training.

--------------------------------------------------------
Installation Instructions
--------------------------------------------------------
1. Install and setup boost (www.boost.org)
2. Download, unzip and copy the "neural_nets" folder into your C++ project folder
3. If using Visual Studio, add the define _SCL_SECURE_NO_WARNINGS to your projects preprocessor symbols 
   ("Project->Properties->Configuration Properties->C/C++->Preprocessor Definitions") in debug mode.
4. You can now include the desired headers and directly use the library

--------------------------------------------------------
Notes/Remarks
--------------------------------------------------------
1. This library was created and tested with boost version 1.57.0
2. For some examples on "how to use", look in the "examples" folder shipped with this library
3. Network serialization isn't fully supported yet: You can output a network to a file, but at the moment not read it 
   in again

--------------------------------------------------------
Frequently Asked Questions (FAQ)
--------------------------------------------------------
Q: What makes this library special over other neural network libraries?
A: It allows arbitrary network structures including shortcut and/or lateral connections as well as
   recurrent backwards and forward connections of arbirary delays. While most available libraries
   for recurrent neural networks only allow delays of one time stamp, this one allows any delay line
   (also containing gaps within one delay line!) between any neuron connection.

Q: Is there a minimal example available on how to use this library?
A: Take a look at the "example_xor.cpp" in the "examples" folder shipped with this library. There a
   static neural network (static means no internal memory) is used for the famous XOR problem.

Q: What does that "train_lm_stepwise" function mean exactly?
A: There are two functions for training, "train_lm" and "train_lm_stepwise". While "train_lm" just
   executes the standard Levenberg-Marquardt optimization, "train_lm_stepwise" performes a more
   advanced training and should usually be the prefered choice. This function executes the standard
   "train_lm" function multiple times with random weight initialization and picks the best result
   of multiple trials. Also you can configure a percentage by which the training signal is segmented.
   The network is then trained on the first data segment which is then stepwise increased until the
   full data is used for training. This is especially useful for training on unstable systems.

Q: I created a network structure, but my code is crashes. What am I doing wrong?
A: A network must be "valid" to be computable. Make sure that you specified input and output neurons,
   and that the input data matches these numbers. Also a network must not contain any algebraic loops
   to be computable. The easiest way to check this is to put your code inside a try-catch block and
   catch all "neural_exceptions". Then outputing the message of the catched exception will show you
   an error message which tells you what went wrong during the computation.

Q: Can I use this library for system identification?
A: Yes, thats one porpuse of this library. Take a look at the "example_recurrent_network.cpp" in the
   "examples" folder shipped with this library. There a first order delay system is excited with an
   optimal APRBS and a neural network is trained to reproduce its output.

Q: What does "optimal APRBS" mean?
A: APRBS stands for "Amplitude modulated Pseudo-Random Binary Sequence". This is a popular excitation signal
   used for system identification, because it approximatly excites all frequencies equally well. Because its
   amplitude is modulated, it can also be used for nonlinear system identification. 

Q: Ok, but what does "optimal" mean in that case?
A: The APRBS generators in this library make "optimal" APRBS in the sense that a user can specify a predefined 
   length and maximum hold time and the APRBS will be matched to these in such a way that their statistical 
   properties are maintained as good as possible.

--------------------------------------------------------
Relevant Header Files
--------------------------------------------------------

There are three headers for the user of this library:

#include "neural_nets\general_net.h"       // General Dynamic Neural Network (GDNN) class template
#include "neural_nets\net_training.h"      // Neural Network training methods (Levenberg-Marquardt)
#include "neural_nets\net_signals.h"       // Optimal APRBS (training signal) generation


As most likely all of those headers are required to do something usefull with the library, there is
also a single header that includes all of those three:

#include "neural_nets\neural_nets.h"       // All relevant headers for full neural network usage