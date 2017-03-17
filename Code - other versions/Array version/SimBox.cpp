#include "SimBox.h"
#include <omp.h>

SimBox::SimBox()
{
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				particles[i][j][k] = new Particle[0];
				num_particles[i][j][k] = 0;

				new_particles[i][j][k] = NULL;
				num_new_particles[i][j][k] = 0;
			}
	//*
	for (int i = -3; i < 4; ++i)
		for (int j = -3; j < 4; ++j)
			for (int k = 0; k < 3; ++k)
			{
				Particle *p_particle = new Particle(Vector3d(0.5 * BOUNDARY_X + 0.05 * i, 0.5 * BOUNDARY_Y + 0.05 * j, 0.05 * k), Vector3d(0.0, 0.0, 0.0));
				put_in_particle(p_particle);
			}
	//*/
}

void SimBox::put_in_particle(Particle *particle)
{
	int x, y, z;
	getGridPosition(particle, x, y, z);
	if (x >= 0 && x < CELL_NUM_X && y >= 0 && y < CELL_NUM_Y && z >= 0 && z < CELL_NUM_Z)
	{
		Particle_node* old_head = new_particles[x][y][z];
		new_particles[x][y][z] = new Particle_node(particle, old_head);
		num_new_particles[x][y][z] += 1;
	}
}

void SimBox::one_tick()
{
//#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
				for (int l = 0; l < num_particles[i][j][k]; ++l)
				{
					particles[i][j][k][l].update_velocity();
					particles[i][j][k][l].update_position();
				}

	updateGrid();

//#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
				for (int l = 0; l < num_particles[i][j][k]; ++l)
				{
					particles[i][j][k][l].calc_density_and_pressure(i, j, k, particles, num_particles);
				}

//#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
				for (int l = 0; l < num_particles[i][j][k]; ++l)
				{
					particles[i][j][k][l].update_acceleration(i, j, k, particles, num_particles);
					particles[i][j][k][l].update_velocity();
				}
}

inline void SimBox::updateGrid()
{
	Particle_node* current_particles[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z];
	int num_current_particles[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z];
//#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				current_particles[i][j][k] = NULL;
				num_current_particles[i][j][k] = 0;
			}

//#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
				for (int l = 0; l < num_particles[i][j][k]; ++l)
				{
					int x, y, z;
					getGridPosition(&(particles[i][j][k][l]), x, y, z);
					if (x >= 0 && x < CELL_NUM_X && y >= 0 && y < CELL_NUM_Y && z >= 0 && z < CELL_NUM_Z)
					{
//#pragma omp critical // Update tmp_particles
						{
							Particle_node* old_head = current_particles[x][y][z];
							current_particles[x][y][z] = new Particle_node(&(particles[i][j][k][l]), old_head);
							num_current_particles[x][y][z] += 1;
						}
					}
					else
						;
				}

	Particle* particles_next[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z];
//#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				particles_next[i][j][k] = new Particle[num_current_particles[i][j][k] + num_new_particles[i][j][k]];

				int l = 0;
				Particle_node* iter;

				iter = new_particles[i][j][k];
				while (iter != NULL)
				{
					particles_next[i][j][k][l] = *(iter->particle);
					++l;
					Particle_node* next = iter->next;
					delete iter->particle;
					delete iter;
					iter = next;
				}

				iter = current_particles[i][j][k];
				while (iter != NULL)
				{
					particles_next[i][j][k][l] = *(iter->particle);
					++l;
					Particle_node* next = iter->next;
					delete iter;
					iter = next;
				}

				num_particles[i][j][k] = num_current_particles[i][j][k] + num_new_particles[i][j][k];
				new_particles[i][j][k] = NULL;
				num_new_particles[i][j][k] = 0;
			}

	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				delete[] particles[i][j][k];
				particles[i][j][k] = particles_next[i][j][k];
			}
}

inline void SimBox::getGridPosition(Particle *particle, int &i, int &j, int &k)
{
	Vector3d position = particle->get_position();
	i = position.x / CELL_SIZE_X;
	j = position.y / CELL_SIZE_Y;
	k = position.z / CELL_SIZE_Z;
}

void SimBox::showScene() {
	// draw particles here
	glColor3f(1.0, 1.0, 1.0);
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
				for (int l = 0; l < num_particles[i][j][k]; ++l)
				{
					glPushMatrix();
					glTranslated(particles[i][j][k][l].get_position().x - 0.5 * BOUNDARY_X, particles[i][j][k][l].get_position().y - 0.5 * BOUNDARY_Y, particles[i][j][k][l].get_position().z);
					glutSolidSphere(0.05, 10, 10);
					glPopMatrix();
				}
}
