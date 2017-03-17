#ifndef FLUID_SIMULATION_PARTICLE
#define FLUID_SIMULATION_PARTICLE

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstddef>
#include "Vector3d.h"

using namespace std;

#define CORE_RADIUS 0.1
#define DELTA_T 0.01

#define BOUNDARY_X 1
#define BOUNDARY_Y 1
#define BOUNDARY_Z 1

#define CELL_NUM_X 11            // num of cells in X-dimension
#define CELL_NUM_Y 11            // num of cells in Y-dimension
#define CELL_NUM_Z 11            // num of cells in Z-dimension
#define CELL_SIZE_X 0.1          // size of a cell in X-dimension
#define CELL_SIZE_Y 0.1          // size of a cell in Y-dimension
#define CELL_SIZE_Z 0.1          // size of a cell in Z-dimension

// CELL_SIZE_i must be >= CORE_RADIUS
// CELL_NUM_i * CELL_SIZE_i must be > BOUNDRY_i

#define M_PI 3.141592653589793238462643383279

struct Particle_node;

class Particle
{
public:
	__device__ Particle(Vector3d mPosition, Vector3d mVelocity) :
		position(mPosition), velocity(mVelocity), acceleration(Vector3d(0.0, 0.0, 0.0)) {}

	__device__ void update_acceleration(int x, int y, int z, Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z]);
	__device__ void update_velocity();
	__device__ void update_position();

	__device__ void calc_density_and_pressure(int x, int y, int z, Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z]);

	__device__ __host__ Vector3d get_position() const { return position; }

private:
	const double mass = 0.001;
	Vector3d position;
	Vector3d velocity;
	Vector3d acceleration;

	double density = 0;
	double pressure = 0;

	__device__ void handleBoundary();
};

struct Particle_node
{
	Particle* particle;
	Particle_node* next;

	__device__ Particle_node(Particle* particle, Particle_node* next) : particle(particle), next(next) {}
};

#endif
