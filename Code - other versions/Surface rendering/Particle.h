#ifndef FLUID_SIMULATION_PARTICLE
#define FLUID_SIMULATION_PARTICLE

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

#define M_PI 3.141592653589793238462643383279

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

// --------------------- End of struct Vector3d ---------------------

struct Particle_node;

class Particle
{
public:
	Particle(Vector3d mPosition, Vector3d mVelocity) :
		position(mPosition), velocity(mVelocity), acceleration(Vector3d(0.0, 0.0, 0.0)) {}

	void update_acceleration(int x, int y, int z, Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z]);
	void update_velocity();
	void update_position();

	void calc_density_and_pressure(int x, int y, int z, Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z]);

	void get_position(double &x, double &y, double &z) const { x = position.x; y = position.y; z = position.z; }
	double get_mass() const { return mass; }
	double get_density() const { return density; }

private:
	const double mass = 0.001;
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

// This array is used to fastly access the neighbor cells of a cell.
const int neighbor_cells[27][3] =
{
	{ -1, -1, -1 },
	{ -1, -1, 0 },
	{ -1, -1, 1 },
	{ -1, 0, -1 },
	{ -1, 0, 0 },
	{ -1, 0, 1 },
	{ -1, 1, -1 },
	{ -1, 1, 0 },
	{ -1, 1, 1 },
	{ 0, -1, -1 },
	{ 0, -1, 0 },
	{ 0, -1, 1 },
	{ 0, 0, -1 },
	{ 0, 0, 0 },
	{ 0, 0, 1 },
	{ 0, 1, -1 },
	{ 0, 1, 0 },
	{ 0, 1, 1 },
	{ 1, -1, -1 },
	{ 1, -1, 0 },
	{ 1, -1, 1 },
	{ 1, 0, -1 },
	{ 1, 0, 0 },
	{ 1, 0, 1 },
	{ 1, 1, -1 },
	{ 1, 1, 0 },
	{ 1, 1, 1 },
};

// ----------------------- Kernel functions -----------------------

inline double W_poly6(double r_x, double r_y, double r_z, double h)
{
	double r2 = r_x * r_x + r_y * r_y + r_z * r_z;
	if (r2 > h * h)
		return 0.0;

	double h2_r2 = h * h - r2;
	double h3 = h * h * h;
	double h9 = h3 * h3 * h3;
	return 315 / (64 * M_PI * h9) * (h2_r2 * h2_r2 * h2_r2);
}

inline void W_poly6_grad(double r_x, double r_y, double r_z, double h, double &res_x, double &res_y, double &res_z)
{
	double r2 = r_x * r_x + r_y * r_y + r_z * r_z;
	if (r2 > h * h)
	{
		res_x = 0.0;
		res_y = 0.0;
		res_z = 0.0;
		return;
	}

	double h2_r2 = h * h - r2;
	double h3 = h * h * h;
	double h9 = h3 * h3 * h3;
	double factor = -945 / (32 * M_PI * h9) * (h2_r2 * h2_r2);
	res_x = factor * r_x;
	res_y = factor * r_y;
	res_z = factor * r_z;
}

inline double W_poly6_Laplacian_modified(double r_x, double r_y, double r_z, double h)
{
	double r2 = r_x * r_x + r_y * r_y + r_z * r_z;
	if (r2 > h * h)
		return 0.0;

	double h2_r2 = h * h - r2;
	double h3 = h * h * h;
	double h9 = h3 * h3 * h3;
	double result = -945 / (32 * M_PI * h9) * (h * h - 7 * r2) * h2_r2;
	if (result < 0)
		return result;
	else
		return 0;
}

inline void W_spiky_grad(double r_x, double r_y, double r_z, double h, double &res_x, double &res_y, double &res_z)
{
	double r_length = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);
	if (r_length > h || r_length == 0.0)
	{
		res_x = 0.0;
		res_y = 0.0;
		res_z = 0.0;
		return;
	}

	double h_r = h - r_length;
	double h3 = h * h * h;
	double h6 = h3 * h3;
	double factor = -45 / (M_PI * h6) * (h_r * h_r) / r_length;
	res_x = factor * r_x;
	res_y = factor * r_y;
	res_z = factor * r_z;
}

inline double W_viscosity_Laplacian(double r_x, double r_y, double r_z, double h)
{
	double r_length = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);
	if (r_length > h)
		return 0.0;

	double h3 = h * h * h;
	double h6 = h3 * h3;
	return 45 / (M_PI * h6) * (h - r_length);
}

// ---------------------- End of kernel functions ----------------------

#endif
