## Kuiperinfer : an inference framework for learners

## Environment Setting

Run Docker_connect.sh to connect to the Docker container. 

Then, connect Visual Studio Code to the Docker container and start working!

:smile::smile::smile::smile::smile::smile:

### Lab1
Use the armadillo library to compute arma::fmat matrix.
A very easy lab, just use easy operation and arma::exp API.(3/30/2024)

### Lab2

Understand Tensor class.(Use template and Armadillo and design user interface. Implemented the float template specialization)
    
Use Armadillo to store the fcube in class and understand the class functions.

Implemented Padding and Flatten. (4/1/2024) 

### Lab3
Learn about deep learning model **intermediate representation**. 

Learn about pnnx(`ir.cpp`, `ir.hpp`. Created by nihui dalao). Learn about model graph, operators and operands structures. 

Learn about kuiperinfer Runtime representation of graph, operators and operands structures. 

By the way, finished CS6290 Project2 yesterday.(4/7/2024)

:guitar: :guitar: :guitar: 

Finish `RuntimeGraph::InitGraphParams();`

Modify the `Runtime_Operator.params` from `std::mapping<std::string, Runtime_Parameter>` to `std::mapping<std::string, std::shared_ptr<Runtime_Parameter>>`, and change the test code. (4/8/2024)

(Just found that I was using another Github account when committing. Change to Pb314314 now.)

:smiley::smiley::smiley::smiley::smiley::smiley::smiley:

### Lab4

Review Lab3, understand `Runtime_Operator` better.

Start Lab4, learn about graph topology path and operator output memory allocation.(4/9/2024)

ALmost finish Lab4. Understand RuntimeGraph::Build and RuntimeGraph::ReverseTopo.

Deeper understand the RuntimeGraph init process and RuntimeOperator class. (4/10/2024)

:musical_note::musical_note::musical_note::musical_note::musical_note:

Finished Lab4, implement the Pb_Topo, using queue instead of recursion to get Topology path. 

By the way, finished CSE6242 HW4.....(4/13/2024)

:sleeping::sleeping::sleeping:

### Lab5
Start Lab5, get to know `Layer` and `Layer_factory`. (4/14/2024)

Half understand the Lab5. Understand **Singular Pattern** in C++.

Use **static in class** which can be used by all instances of that class, which is a implementation of **thread-safe singular pattern**.

Use gdb to test program.(cmake -DCMAKE_BUILD_TYPE=Debug ..)

Understand the process of register operator, prepare input and compute output.(4/15/2024)

Add a google test and better understand registration and layer intialization. Add `sigmoid.hpp`, `sigmoid.cpp` files. 

Finish registration of Sigmoid. Only thing left for Lab5 is the sigmoid forward function.(4/16/2024)

Finish Sigmoid operator. (4/17/2024)

:airplane::airplane::airplane:

### Lab6
Start Lab6. Understand the general data flow of convolution.cpp. Generally go throught the GetInstance and Forward functions. (4/19/2024)

Read almost all the code from course6, but not fully understand computing process. The data organization still vague to me. Confuse about conv2d Im2Col data organization.(4/24/2024)

### Lab7
Start Lab7. Understand how to convert expression string into syntax tree. 
First, get statement_ string. ie. `add(mul(@0,@1),@2)`. Split string into strings and tokens.
Use generate() function to create syntax tree.

:sleepy::sleepy::sleepy:

Almost finish Lab7. Read the Expression layer code. Need to finish sin Forward function tonight.(4/26/2024)

### Lab8
Start Lab8. Read through the Graph Forward function. Understand the Resnet forward process. Need to read through other operators' code.(4/27/2024)

### Lab9

