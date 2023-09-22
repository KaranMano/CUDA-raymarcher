#pragma once
#include "vector.h"
#include <limits.h>
#include <vector>
#include <cuda_runtime.h>
#include <memory>

class Object;

class Ray {
private:
	Vector m_origin, m_direction;
	int m_interected;
	float m_hit;

public:

	__host__ __device__
	Ray();
	__host__ __device__
	Ray(const Vector &_origin, const Vector &_direction);

	__host__ __device__
	float param() const;
	__host__ __device__
	Vector hit() const;
	__host__ __device__
	void hit(float _hit, int object);
	__host__ __device__
	void reset();
	__host__ __device__
	const Vector& origin() const;
	__host__ __device__
	void origin(const Vector& v);
	__host__ __device__
	const Vector& direction() const;
	__host__ __device__
	int intersected() const;
};