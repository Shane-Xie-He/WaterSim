#ifndef FLUID_SIMULATION_VECTOR3D
#define FLUID_SIMULATION_VECTOR3D

#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct Vector3d
{
	double x;
	double y;
	double z;

	__device__ Vector3d() {}
	__device__ Vector3d(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
};

__device__ double length(const Vector3d &vec);

__device__ Vector3d operator+(const Vector3d &vec1, const Vector3d &vec2);

__device__ Vector3d operator-(const Vector3d &vec1, const Vector3d &vec2);

__device__ Vector3d operator*(const Vector3d &vec, double scalar);

__device__ Vector3d operator/(const Vector3d &vec, double scalar);

__device__ Vector3d operator*(double scalar, const Vector3d &vec);

#endif
