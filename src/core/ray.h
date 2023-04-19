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
	std::shared_ptr<Object> m_interected;
	float m_hit;

public:

	Ray();
	Ray(const Vector &_origin, const Vector &_direction);

	float param() const;
	Vector hit() const;
	void hit(float _hit, const std::shared_ptr<Object>& object);
	void reset();
	const Vector& origin() const;
	void origin(const Vector& v);
	const Vector& direction() const;
	const std::shared_ptr<Object>& intersected() const;
};