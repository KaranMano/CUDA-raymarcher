#pragma once
#include <cuda_runtime.h>

class Vector {
	public:
		float x, y, z;
		__host__ __device__ Vector();
		__host__ __device__ Vector(const float &_x, const float &_y, const float &_z);
		__host__ __device__ Vector(const Vector &v);

		__host__ __device__ Vector operator-() const;
		__host__ __device__ Vector& operator+=(const Vector& rhs);
		__host__ __device__ Vector& operator-=(const Vector& rhs);
		__host__ __device__ Vector& operator/=(const Vector& rhs);
		__host__ __device__ Vector& operator*=(const Vector& rhs);
		__host__ __device__ Vector& operator*=(float scalar);

		__host__ __device__ Vector& operator=(const Vector& rhs);
		__host__ __device__ Vector operator+(const Vector& rhs) const;
		__host__ __device__ Vector operator-(const Vector& rhs) const;
		__host__ __device__ Vector operator*(const Vector& rhs) const;
		__host__ __device__ Vector operator/(const Vector& rhs) const;
		
		__host__ __device__ Vector operator*(float scalar) const;
		__host__ __device__ friend Vector operator*(float scalar, const Vector& rhs);
};

__host__ __device__ Vector cross(const Vector& lhs, const Vector &rhs);
__host__ __device__ float dot(const Vector& lhs, const Vector &rhs);