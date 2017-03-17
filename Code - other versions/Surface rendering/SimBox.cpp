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
	double position_x, position_y, position_z;
	particle->get_position(position_x, position_y, position_z);
	i = position_x / CELL_SIZE_X;
	j = position_y / CELL_SIZE_Y;
	k = position_z / CELL_SIZE_Z;
}

void SimBox::showScene() {
	/*
	// draw particles here
	for (int i = 0; i < CELL_NUM_X; ++i)
		for (int j = 0; j < CELL_NUM_Y; ++j)
			for (int k = 0; k < CELL_NUM_Z; ++k)
			{
				Particle_node* iter = cells[i][j][k];
				while (iter != NULL)
				{
					glPushMatrix();
					double position[3];
					iter->particle->get_position(position[0], position[1], position[2]);
					glTranslated(position[0], position[1], position[2]);
					glutSolidSphere(0.05, 10, 10);
					glPopMatrix();
					iter = iter->next;
				}
			}
	//*/
	//*
	const double edge = 0.05;
	for (int i = -2; i < int(CELL_NUM_X * CELL_SIZE_X / edge) + 1; ++i)
		for (int j = -2; j < int(CELL_NUM_Y * CELL_SIZE_Y / edge) + 1; ++j)
			for (int k = -2; k < int(CELL_NUM_Z * CELL_SIZE_Z / edge) + 1; ++k)
				cube_draw(i * edge, j * edge, k * edge, edge);
	//*/
}

inline double vertex_find_position(double first_value, double second_value)
{
	return (0.2 - first_value) / (second_value - first_value);
	//return 0.5;
}

inline void SimBox::cube_draw(double pos_x, double pos_y, double pos_z, double edge)
{
	Vector3d vertices[8] = {
		{ pos_x, pos_y, pos_z },
		{ pos_x + edge, pos_y, pos_z },
		{ pos_x + edge, pos_y + edge, pos_z },
		{ pos_x, pos_y + edge, pos_z },
		{ pos_x, pos_y, pos_z + edge },
		{ pos_x + edge, pos_y, pos_z + edge },
		{ pos_x + edge, pos_y + edge, pos_z + edge },
		{ pos_x, pos_y + edge, pos_z + edge }
	};

	double vertex_color[8];
	int cube_index = 0;
	for (int iii = 0; iii < 8; ++iii)
	{
		int x = vertices[iii].x / CELL_SIZE_X;
		int y = vertices[iii].y / CELL_SIZE_Y;
		int z = vertices[iii].z / CELL_SIZE_Z;

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

				double that_pos_x, that_pos_y, that_pos_z;
				that->get_position(that_pos_x, that_pos_y, that_pos_z);
				double W_c = W_poly6(vertices[iii].x - that_pos_x, vertices[iii].y - that_pos_y, vertices[iii].z - that_pos_z, CORE_RADIUS);
				sum = sum + that->get_mass() / that->get_density() * W_c;

				iter = iter->next;
			}
		}
		vertex_color[iii] = sum;
		if (sum > 0.2)
			cube_index |= (1 << iii);
	}

	Vector3d triVert[16];
	Vector3d triVertNorm[16];
	for (int i = 0; i < 16; ++i)
	{
		int edge_num = triTable[cube_index][i];
		Vector3d vertex;

		if (edge_num == -1)
			break;

		else if (edge_num == 0)
			vertex = Vector3d(vertex_find_position(vertex_color[0], vertex_color[1]) * edge, 0.0, 0.0);
		else if (edge_num == 1)
			vertex = Vector3d(edge, vertex_find_position(vertex_color[1], vertex_color[2]) * edge, 0.0);
		else if (edge_num == 2)
			vertex = Vector3d(vertex_find_position(vertex_color[3], vertex_color[2]) * edge, edge, 0.0);
		else if (edge_num == 3)
			vertex = Vector3d(0.0, vertex_find_position(vertex_color[0], vertex_color[3]) * edge, 0.0);

		else if (edge_num == 4)
			vertex = Vector3d(vertex_find_position(vertex_color[4], vertex_color[5]) * edge, 0.0, edge);
		else if (edge_num == 5)
			vertex = Vector3d(edge, vertex_find_position(vertex_color[5], vertex_color[6]) * edge, edge);
		else if (edge_num == 6)
			vertex = Vector3d(vertex_find_position(vertex_color[7], vertex_color[6]) * edge, edge, edge);
		else if (edge_num == 7)
			vertex = Vector3d(0.0, vertex_find_position(vertex_color[4], vertex_color[7]) * edge, edge);

		else if (edge_num == 8)
			vertex = Vector3d(0.0, 0.0, vertex_find_position(vertex_color[0], vertex_color[4]) * edge);
		else if (edge_num == 9)
			vertex = Vector3d(edge, 0.0, vertex_find_position(vertex_color[1], vertex_color[5]) * edge);
		else if (edge_num == 10)
			vertex = Vector3d(edge, edge, vertex_find_position(vertex_color[2], vertex_color[6]) * edge);
		else if (edge_num == 11)
			vertex = Vector3d(0.0, edge, vertex_find_position(vertex_color[3], vertex_color[7]) * edge);

		vertex.x += pos_x;
		vertex.y += pos_y;
		vertex.z += pos_z;

		triVert[i] = vertex;

		int x = vertex.x / CELL_SIZE_X;
		int y = vertex.y / CELL_SIZE_Y;
		int z = vertex.z / CELL_SIZE_Z;

		double sum[3] = { 0.0, 0.0, 0.0 };
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

				double that_pos_x, that_pos_y, that_pos_z;
				that->get_position(that_pos_x, that_pos_y, that_pos_z);
				double W_grad_c[3];
				W_poly6_grad(vertex.x - that_pos_x, vertex.y - that_pos_y, vertex.z - that_pos_z, CORE_RADIUS, W_grad_c[0], W_grad_c[1], W_grad_c[2]);
				double factor = that->get_mass() / that->get_density();
				sum[0] += factor * W_grad_c[0];
				sum[1] += factor * W_grad_c[1];
				sum[2] += factor * W_grad_c[2];

				iter = iter->next;
			}
		}
		
		double length_sum = sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
		triVertNorm[i].x = -sum[0] / length_sum;
		triVertNorm[i].y = -sum[1] / length_sum;
		triVertNorm[i].z = -sum[2] / length_sum;
	}

	glBegin(GL_TRIANGLES);
	for (int i = 0; i < 16; ++i)
	{
		int edge_num = triTable[cube_index][i];
		if (edge_num == -1)
			break;

		glNormal3d(triVertNorm[i].x, triVertNorm[i].y, triVertNorm[i].z);
		glVertex3d(triVert[i].x, triVert[i].y, triVert[i].z);
	}
	glEnd();
}
