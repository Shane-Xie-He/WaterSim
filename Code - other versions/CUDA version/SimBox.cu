#include "SimBox.h"

dim3 size = { CELL_NUM_X, CELL_NUM_Y, CELL_NUM_Z };
cudaError_t cudaStatus;

SimBox::SimBox()
{
	cudaStatus = cudaMalloc(&num_particles, sizeof(int));
	cudaStatus = cudaMemset(num_particles, 0, sizeof(int));

	cudaStatus = cudaMalloc(&cells, sizeof(Particle_node*) * CELL_NUM_X * CELL_NUM_Y * CELL_NUM_Z);
	cudaStatus = cudaMemset(cells, NULL, sizeof(Particle_node*) * CELL_NUM_X * CELL_NUM_Y * CELL_NUM_Z);
	//*
	for (int i = -3; i < 4; ++i)
		for (int j = -3; j < 4; ++j)
			for (int k = 0; k < 3; ++k)
			{
				put_in_particle(0.5 * BOUNDARY_X + 0.05 * i, 0.5 * BOUNDARY_Y + 0.05 * j, 0.05 * k, 0.0, 0.0, 0.0);
			}
	//*/
}

__device__ void getGridPosition(Particle *particle, int &i, int &j, int &k)
{
	Vector3d position = particle->get_position();
	i = position.x / CELL_SIZE_X;
	j = position.y / CELL_SIZE_Y;
	k = position.z / CELL_SIZE_Z;
}

__global__ void new_particle(double r_x, double r_y, double r_z, double v_x, double v_y, double v_z, Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z], int* num_particles)
{
	Particle* particle = new Particle(Vector3d(r_x, r_y, r_z), Vector3d(v_x, v_y, v_z));
	int x, y, z;
	getGridPosition(particle, x, y, z);
	if (x >= 0 && x < CELL_NUM_X && y >= 0 && y < CELL_NUM_Y && z >= 0 && z < CELL_NUM_Z)
	{
		Particle_node* old_head = cells[x][y][z];
		cells[x][y][z] = new Particle_node(particle, old_head);
		*num_particles += 1;
	}
	else
		delete particle;
}

void SimBox::put_in_particle(double r_x, double r_y, double r_z, double v_x, double v_y, double v_z)
{
	new_particle<<<1, 1>>>(r_x, r_y, r_z, v_x, v_y, v_z, cells, num_particles);
	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();
}

__global__ void one_tick_1(Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z])
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;
	Particle_node* iter = cells[i][j][k];
	while (iter != NULL)
	{
		iter->particle->update_velocity();
		iter->particle->update_position();
		iter = iter->next;
	}
}

__global__ void one_tick_2(Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z])
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;
	Particle_node* iter = cells[i][j][k];
	while (iter != NULL)
	{
		iter->particle->calc_density_and_pressure(i, j, k, cells);
		iter = iter->next;
	}
}

__global__ void one_tick_3(Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z])
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;
	Particle_node* iter = cells[i][j][k];
	while (iter != NULL)
	{
		iter->particle->update_acceleration(i, j, k, cells);
		iter->particle->update_velocity();
		iter = iter->next;
	}
}

void SimBox::one_tick()
{
	one_tick_1<<<size, 1>>>(cells);
	cudaDeviceSynchronize();

	updateGrid();

	one_tick_2<<<size, 1>>>(cells);
	cudaDeviceSynchronize();

	one_tick_3<<<size, 1>>>(cells);
	cudaDeviceSynchronize();
}

__global__ void updateGrid_1(Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z], Particle_node* tmp_cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z], int* num_particles)
{
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
							Particle_node* old_head = tmp_cells[x][y][z];
							tmp_cells[x][y][z] = *iter;
							tmp_cells[x][y][z]->next = old_head;
						}
						else
						{
							delete (*iter)->particle;
							delete (*iter);
							*num_particles -= 1;
						}
						*iter = next;
					}
					else
						iter = &((*iter)->next);
				}
			}
}

__global__ void updateGrid_2(Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z], Particle_node* tmp_cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z])
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;
	Particle_node** iter = &(cells[i][j][k]);
	while (*iter != NULL)
		iter = &((*iter)->next);
	*iter = tmp_cells[i][j][k];
}

inline void SimBox::updateGrid()
{
	Particle_node* (*tmp_cells)[CELL_NUM_Y][CELL_NUM_Z];
	cudaMalloc(&tmp_cells, sizeof(Particle_node*) * CELL_NUM_X * CELL_NUM_Y * CELL_NUM_Z);
	cudaMemset(tmp_cells, NULL, sizeof(Particle_node*) * CELL_NUM_X * CELL_NUM_Y * CELL_NUM_Z);

	updateGrid_1<<<1, 1>>>(cells, tmp_cells, num_particles);
	cudaDeviceSynchronize();

	updateGrid_2<<<size, 1>>>(cells, tmp_cells);
	cudaDeviceSynchronize();
}

__global__ void get_all_positions(Particle_node* cells[CELL_NUM_X][CELL_NUM_Y][CELL_NUM_Z], double(*positions)[3])
{
	int ii = 0;
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				Particle_node* iter = cells[i][j][k];
				while (iter != NULL)
				{
					positions[ii][0] = iter->particle->get_position().x;
					positions[ii][1] = iter->particle->get_position().y;
					positions[ii][2] = iter->particle->get_position().z;
					ii++;
					iter = iter->next;
				}
			}
}

void SimBox::showScene() {
	int host_num_particles;
	cudaMemcpy(&host_num_particles, num_particles, sizeof(int), cudaMemcpyDeviceToHost);

	double(*dev_positions)[3];
	cudaMalloc(&dev_positions, sizeof(double) * host_num_particles * 3);

	get_all_positions<<<1, 1>>>(cells, dev_positions);
	cudaDeviceSynchronize();

	double(*positions)[3] = (double(*)[3])malloc(sizeof(double) * host_num_particles * 3);
	cudaMemcpy(positions, dev_positions, sizeof(double) * host_num_particles * 3, cudaMemcpyDeviceToHost);

	for (int i = 0; i < host_num_particles; ++i)
	{
		glPushMatrix();
		glTranslated(positions[i][0] - 0.5 * BOUNDARY_X, positions[i][1] - 0.5 * BOUNDARY_Y, positions[i][2]);
		glutSolidSphere(0.05, 10, 10);
		glPopMatrix();
	}

	cudaFree(dev_positions);
	free(positions);
}
