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
	void put_in_particle(Particle *particle);
	// one tick of the clock
	void one_tick();

private:
	inline void updateGrid();
	inline void getGridPosition(Particle *particle, int &i, int &j, int &k);

	Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z];
};

#endif
