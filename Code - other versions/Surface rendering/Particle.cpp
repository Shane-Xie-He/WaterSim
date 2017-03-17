#include "Particle.h"

void Particle::update_acceleration(int x, int y, int z, Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z])
{
	double sum_p[3] = { 0.0, 0.0, 0.0 };
	double sum_v[3] = { 0.0, 0.0, 0.0 };
	double sum_s[3] = { 0.0, 0.0, 0.0 };

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

			double W_grad_p[3];
			W_spiky_grad(position.x - that->position.x, position.y - that->position.y, position.z - that->position.z, CORE_RADIUS, W_grad_p[0], W_grad_p[1], W_grad_p[2]);
			double factor_p = that->mass * ((pressure + that->pressure) / (2 * that->density));
			sum_p[0] += factor_p * W_grad_p[0];
			sum_p[1] += factor_p * W_grad_p[1];
			sum_p[2] += factor_p * W_grad_p[2];

			double W_Laplacian_v = W_viscosity_Laplacian(position.x - that->position.x, position.y - that->position.y, position.z - that->position.z, CORE_RADIUS);
			double factor_v = that->mass / that->density * W_Laplacian_v;
			sum_v[0] += factor_v * (that->velocity.x - velocity.x);
			sum_v[1] += factor_v * (that->velocity.y - velocity.y);
			sum_v[2] += factor_v * (that->velocity.z - velocity.z);

			double W_s = W_poly6(position.x - that->position.x, position.y - that->position.y, position.z - that->position.z, CORE_RADIUS);
			double factor_s = that->mass / that->density * W_s;
			sum_s[0] += factor_s * (that->position.x - position.x);
			sum_s[1] += factor_s * (that->position.y - position.y);
			sum_s[2] += factor_s * (that->position.z - position.z);

			iter = iter->next;
		}
	}

	const double viscosity = 0.1;
	const double surface_tension = 10;

	acceleration.x = (-sum_p[0] + viscosity * sum_v[0] + surface_tension * sum_s[0]) / density;
	acceleration.y = (-sum_p[1] + viscosity * sum_v[1] + surface_tension * sum_s[1]) / density;
	acceleration.z = (-sum_p[2] + viscosity * sum_v[2] + surface_tension * sum_s[2]) / density - 9.8;
}

void Particle::update_velocity()
{
	velocity.x = velocity.x + acceleration.x * (DELTA_T / 2);
	velocity.y = velocity.y + acceleration.y * (DELTA_T / 2);
	velocity.z = velocity.z + acceleration.z * (DELTA_T / 2);
}

void Particle::update_position()
{
	position.x = position.x + velocity.x * DELTA_T;
	position.y = position.y + velocity.y * DELTA_T;
	position.z = position.z + velocity.z * DELTA_T;
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

			double W_d = W_poly6(position.x - that->position.x, position.y - that->position.y, position.z - that->position.z, CORE_RADIUS);
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
