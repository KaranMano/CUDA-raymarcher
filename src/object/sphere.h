#pragma once
#include "object.h"
#include "../core/vector.h"
#include "../material/material.h"
#include <iostream>

class Sphere : public Object {
private:
	float m_radius;
public:
	__host__ __device__
		Sphere(const Vector &_position, float _radius, bool _volume);
	__host__ __device__
		float intersect(const Ray& ray) const override;
	__host__ __device__
		Vector normal(const Vector &point) const override;
	__host__ __device__
		float radius();
};
