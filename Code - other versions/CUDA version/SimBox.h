#ifndef FLUID_SIMULATION_SIMBOX
#define FLUID_SIMULATION_SIMBOX

#include <stdlib.h>
#include <GL/glut.h>
#include "Particle.h"

#define PI 3.141592653589793238462643383279

class SimBox
{
public:
	SimBox();

	// show scene using computed parameters: acceleration, velocity, position...
	void showScene();
	void put_in_particle(double r_x, double r_y, double r_z, double v_x, double v_y, double v_z);
	// one tick of the clock
	void one_tick();

private:
	inline void updateGrid();

	Particle_node* (*cells)[CELL_NUM_Y][CELL_NUM_Z];
	// This is a 3-dimenstional array of type "Particle_node*" that is stored on GPU
	int* num_particles; // This number is stored on GPU
};

#endif
