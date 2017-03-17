#include "Vector3d.h"

__device__ double length(const Vector3d &vec)
{
	return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__ Vector3d operator+(const Vector3d &vec1, const Vector3d &vec2)
{
	return Vector3d(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z);
}

__device__ Vector3d operator-(const Vector3d &vec1, const Vector3d &vec2)
{
	return Vector3d(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z);
}

__device__ Vector3d operator*(const Vector3d &vec, double scalar)
{
	return Vector3d(vec.x * scalar, vec.y * scalar, vec.z * scalar);
}

__device__ Vector3d operator/(const Vector3d &vec, double scalar)
{
	return Vector3d(vec.x / scalar, vec.y / scalar, vec.z / scalar);
}

__device__ Vector3d operator*(double scalar, const Vector3d &vec)
{
	return Vector3d(scalar * vec.x, scalar * vec.y, scalar * vec.z);
}
