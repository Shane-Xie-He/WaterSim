#include "SimBox.h"
#include <omp.h>

SimBox::SimBox()
{
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
				cells[i][j][k] = NULL;
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
		Particle_node* old_head = cells[x][y][z];
		cells[x][y][z] = new Particle_node(particle, old_head);
	}
}

void SimBox::one_tick()
{
#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				Particle_node* iter = cells[i][j][k];
				while (iter != NULL)
				{
					iter->particle->update_velocity();
					iter->particle->update_position();
					iter = iter->next;
				}
			}

	updateGrid();

#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				Particle_node* iter = cells[i][j][k];
				while (iter != NULL)
				{
					iter->particle->calc_density_and_pressure(i, j, k, cells);
					iter = iter->next;
				}
			}

#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				Particle_node* iter = cells[i][j][k];
				while (iter != NULL)
				{
					iter->particle->update_acceleration(i, j, k, cells);
					iter->particle->update_velocity();
					iter = iter->next;
				}
			}
}

inline void SimBox::updateGrid()
{
	Particle_node* tmp_cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z];
#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
				tmp_cells[i][j][k] = NULL;

#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				Particle_node** iter = &(cells[i][j][k]);
				while (*iter != NULL)
				{
					int x, y, z;
					getGridPosition((*iter)->particle, x, y, z);
					if (x != i || y != j || z != k)
					{
						Particle_node* next = (*iter)->next;
						if (x >= 0 && x < CELL_NUM_X && y >= 0 && y < CELL_NUM_Y && z >= 0 && z < CELL_NUM_Z)
						{
#pragma omp critical // Update tmp_cells
							{
								Particle_node* old_head = tmp_cells[x][y][z];
								tmp_cells[x][y][z] = *iter;
								tmp_cells[x][y][z]->next = old_head;
							}
						}
						else
						{
							delete (*iter)->particle;
							delete (*iter);
						}
						*iter = next;
					}
					else
						iter = &((*iter)->next);
				}
			}

#pragma omp parallel for
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				Particle_node** iter = &(cells[i][j][k]);
				while (*iter != NULL)
					iter = &((*iter)->next);
				*iter = tmp_cells[i][j][k];
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
			{
				Particle_node* iter = cells[i][j][k];
				while (iter != NULL)
				{
					glPushMatrix();
					glTranslated(iter->particle->get_position().x - 0.5 * BOUNDARY_X, iter->particle->get_position().y - 0.5 * BOUNDARY_Y, iter->particle->get_position().z);
					glutSolidSphere(0.05, 10, 10);
					glPopMatrix();
					iter = iter->next;
				}
			}

}
