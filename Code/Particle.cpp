#include "Particle.h"

// ----------------------- Kernel functions -----------------------

inline double W_poly6(const Vector3d &r, double h)
{
	double r2 = r.x * r.x + r.y * r.y + r.z * r.z;
	if (r2 > h * h)
		return 0.0;

	double h2_r2 = h * h - r2;
	return 315 / (64 * M_PI * pow(h, 9)) * (h2_r2 * h2_r2 * h2_r2);
}

inline Vector3d W_poly6_grad(const Vector3d &r, double h)
{
	double r2 = r.x * r.x + r.y * r.y + r.z * r.z;
	if (r2 > h * h)
		return Vector3d(0.0, 0.0, 0.0);

	double h2_r2 = h * h - r2;
	return (-945 / (32 * M_PI * pow(h, 9)) * (h2_r2 * h2_r2)) * r;
}

inline double W_poly6_Laplacian_modified(const Vector3d &r, double h)
{
	double r2 = r.x * r.x + r.y * r.y + r.z * r.z;
	if (r2 > h * h)
		return 0.0;

	double h2_r2 = h * h - r2;
	double result = -945 / (32 * M_PI * pow(h, 9)) * (h * h - 7 * r2) * h2_r2;
	if (result < 0)
		return result;
	else
		return 0;
}

inline Vector3d W_spiky_grad(const Vector3d &r, double h)
{
	double r_length = length(r);
	if (r_length > h)
		return Vector3d(0.0, 0.0, 0.0);
	if (r_length == 0.0)
		return Vector3d(0.0, 0.0, 0.0);

	double h_r = h - r_length;
	return (-45 / (M_PI * pow(h, 6)) * (h_r * h_r) / r_length) * r;
}

inline double W_viscosity_Laplacian(const Vector3d &r, double h)
{
	double r_length = length(r);
	if (r_length > h)
		return 0.0;

	return 45 / (M_PI * pow(h, 6)) * (h - r_length);
}

// ---------------------- End of kernel functions ----------------------

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

void Particle::update_acceleration(int x, int y, int z, Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z])
{
	Vector3d sum_p = Vector3d(0.0, 0.0, 0.0);
	Vector3d sum_v = Vector3d(0.0, 0.0, 0.0);
	//Vector3d sum_n = Vector3d(0.0, 0.0, 0.0);
	Vector3d sum_s = Vector3d(0.0, 0.0, 0.0);

	for (int ii = 0; ii < 27; ++ii)
	{
		int i = x + neighbor_cells[ii][0];
		int j = y + neighbor_cells[ii][1];
		int k = z + neighbor_cells[ii][2];
		if (i < 0 || i >= CELL_NUM_X)
			continue;
		if (j < 0 || j >= CELL_NUM_Y)
			continue;
		if (k < 0 || k >= CELL_NUM_Z)
			continue;

		Particle_node* iter = cells[i][j][k];
		while (iter != NULL)
		{
			Particle* that = iter->particle;

			Vector3d W_grad_p = W_spiky_grad(position - that->position, CORE_RADIUS);
			sum_p = sum_p + that->mass * ((pressure + that->pressure) / (2 * that->density)) * W_grad_p;

			double W_Laplacian_v = W_viscosity_Laplacian(position - that->position, CORE_RADIUS);
			sum_v = sum_v + that->mass * (that->velocity - velocity) / that->density * W_Laplacian_v;

			//Vector3d W_grad_s = W_poly6_grad(position - that->position, CORE_RADIUS);
			//sum_n = sum_n + that->mass / that->density * W_grad_s;

			double W_s = W_poly6(position - that->position, CORE_RADIUS);
			sum_s = sum_s + that->mass * (that->position - position) / that->density * W_s;

			iter = iter->next;
		}
	}

	Vector3d f_pressure = Vector3d(0.0, 0.0, 0.0) - sum_p;

	const double viscosity = 0.1;
	Vector3d f_viscosity = viscosity * sum_v;

	Vector3d f_surface = 10 * sum_s;

	/*
	Vector3d n = sum_n;
	double n_length = length(n);
	
	Vector3d f_surface;
	if (n_length > 1)
	{
		double sum_s = 0.0;
		for (int ii = 0; ii < 27; ++ii)
		{
			int i = x + neighbor_cells[ii][0];
			int j = y + neighbor_cells[ii][1];
			int k = z + neighbor_cells[ii][2];
			if (i < 0 || i >= CELL_NUM_X)
				continue;
			if (j < 0 || j >= CELL_NUM_Y)
				continue;
			if (k < 0 || k >= CELL_NUM_Z)
				continue;

			Particle_node* iter = cells[i][j][k];
			while (iter != NULL)
			{
				Particle* that = iter->particle;

				double W_Laplacian_s = W_poly6_Laplacian_modified(position - that->position, CORE_RADIUS);
				sum_s = sum_s + that->mass / that->density * W_Laplacian_s;

				iter = iter->next;
			}
		}
		const double tension_coeff = 1.0; // Tension coefficient of water-air interface
		f_surface = (-tension_coeff * sum_s / n_length) * n;
	}
	else
		f_surface = Vector3d(0.0, 0.0, 0.0);
	//*/

	acceleration = (f_pressure + f_viscosity + f_surface) / density + Vector3d(0.0, 0.0, -9.8);
}

void Particle::update_velocity()
{
	velocity = velocity + acceleration * (DELTA_T / 2);
}

void Particle::update_position()
{
	position = position + velocity * DELTA_T;
	handleBoundary();
}

void Particle::calc_density_and_pressure(int x, int y, int z, Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z])
{
	double sum = 0.0;
	for (int ii = 0; ii < 27; ++ii)
	{
		int i = x + neighbor_cells[ii][0];
		int j = y + neighbor_cells[ii][1];
		int k = z + neighbor_cells[ii][2];
		if (i < 0 || i >= CELL_NUM_X)
			continue;
		if (j < 0 || j >= CELL_NUM_Y)
			continue;
		if (k < 0 || k >= CELL_NUM_Z)
			continue;

		Particle_node* iter = cells[i][j][k];
		while (iter != NULL)
		{
			Particle* that = iter->particle;

			double W_d = W_poly6(position - that->position, CORE_RADIUS);
			sum = sum + that->mass * W_d;

			iter = iter->next;
		}
	}
	density = sum;

	pressure = 10 * density;
}

inline void Particle::handleBoundary() {
	if (position.x < 0) {
		position.x = 0;
		velocity.x = -velocity.x;
	}
	else if (position.x > BOUNDARY_X) {
		position.x = BOUNDARY_X;
		velocity.x = -velocity.x;
	}

	if (position.y < 0) {
		position.y = 0;
		velocity.y = -velocity.y;
	}
	else if(position.y > BOUNDARY_Y) {
		position.y = BOUNDARY_Y;
		velocity.y = -velocity.y;
	}

	if (position.z < 0) {
		position.z = 0;
		velocity.z = -velocity.z;
	}
	else if (position.z > BOUNDARY_Z) {
		position.z = BOUNDARY_Z;
		velocity.z = -velocity.z;
	}
}
