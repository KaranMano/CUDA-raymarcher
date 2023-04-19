#pragma once
#include "object.h"
#include "../core/vector.h"
#include "../material/material.h"
#include <iostream>

class Sphere : public Object {
private:
	float m_radius;
public:
	 Sphere(const Vector &_position, float _radius, const std::shared_ptr<Material>& _material, bool _volume);
	 float intersect(const Ray& ray) const override;
	 Vector normal(const Vector &point) const override;
	 float radius();
};
