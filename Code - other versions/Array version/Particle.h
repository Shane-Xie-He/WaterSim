#ifndef FLUID_SIMULATION_PARTICLE
#define FLUID_SIMULATION_PARTICLE

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstddef>

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

// -------------------------- struct Vector3d --------------------------

struct Vector3d
{
	double x;
	double y;
	double z;

	Vector3d() {}
	Vector3d(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
};

inline double length(const Vector3d &vec)
{
	return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

inline Vector3d operator+(const Vector3d &vec1, const Vector3d &vec2)
{
	return Vector3d(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z);
}

inline Vector3d operator-(const Vector3d &vec1, const Vector3d &vec2)
{
	return Vector3d(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z);
}

inline Vector3d operator*(const Vector3d &vec, double scalar)
{
	return Vector3d(vec.x * scalar, vec.y * scalar, vec.z * scalar);
}

inline Vector3d operator/(const Vector3d &vec, double scalar)
{
	return Vector3d(vec.x / scalar, vec.y / scalar, vec.z / scalar);
}

inline Vector3d operator*(double scalar, const Vector3d &vec)
{
	return Vector3d(scalar * vec.x, scalar * vec.y, scalar * vec.z);
}

// --------------------- End of struct Vector3d ---------------------

struct Particle_node;

class Particle
{
public:
	Particle() {}
	Particle(Vector3d mPosition, Vector3d mVelocity) :
		mass(0.001), position(mPosition), velocity(mVelocity), acceleration(Vector3d(0.0, 0.0, 0.0)) {}

	void update_acceleration(int x, int y, int z, Particle* particles[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z], int num_particles[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z]);
	void update_velocity();
	void update_position();

	void calc_density_and_pressure(int x, int y, int z, Particle* particles[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z], int num_particles[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z]);

	Vector3d get_position() const { return position; }

private:
	double mass;
	Vector3d position;
	Vector3d velocity;
	Vector3d acceleration;

	double density = 0;
	double pressure = 0;

	inline void handleBoundary();
};

struct Particle_node
{
	Particle* particle;
	Particle_node* next;

	Particle_node(Particle* particle, Particle_node* next) : particle(particle), next(next) {}
};

#endif
