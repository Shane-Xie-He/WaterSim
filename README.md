# WaterSim

This a particle-based fluid simulation project. It was a course project for Fall 2015 CSCI-596 "Scientific Computing and Simulation" at USC.

We use a particle-based model to simulate the motion of fluid, such as water, based on the Navier-Stocks equations from fluid mechanics. For detailed explanation of our implementation, please see our [project report](https://github.com/Shane-Xie-He/WaterSim/raw/master/Report.pdf).

## Guide for Compiling and Running

We use GLEW 1.13.0 and GLUT 3.7 in our program.

Our program contains OpenMP directives, with OpenMP support our program can have better performance.

When compiling our CUDA code, be sure to tell the compiler to “generate relocatable device code”.

When running our program, use W key and S key to zoom in and zoom out, and use the right button of the mouse to drag the scene to rotate it.

Also you can use the space key to start/stop saving each frame as a picture.

## Brief code explanation

This explanation is based on the original version of our code in “Code/”.

We model particles as a class, defined in “Particle.h” and “Particle.cpp”. We also treat the simulation box as a class, defined in “SimBox.h” and “SimBox.cpp”.

These four files may contain other contents. For example, “struct Vector3d” is a structure in which we store 3-dimensional vectors, and “struct Particle_node” is a structure that we use as a node in the linked lists that store particles.

“main.cpp” is the entrance of the program and its code is structured in GLUT’s way.

“pic.h”, “pic.cpp” and “ppm.cpp” are used to provide the function of saving each frame as a picture. These three files are not written by us.

In the CUDA version of the code, the “struct Vector3d” and its related functions are separately defined in “Vector3d.h” and “Vector3d.cu”, like a formal C++ class.

In the “Surface rendering” version, compared to other versions, we try to use “Vector3d” less, and we also remove the +, -, *, / functions of “Vector3d” and implement these operations directly in the lines where they are used, in order to get some speed-up.
