#pragma once
#include "../material/material.h"
#include <cmath>
#include <random>
class Volume : public Material {
private:
	Vector m_color;
	int m_p[512];
public:
	Volume();
	Volume(const Vector &_color);
	Volume(const Volume  &other);

	float density(const Vector& point) const;
	float phase(float g, float cos) const;
	float noise(const Vector &v) const;
	Vector shade(Ray& ray, const Object& object, const Scene& scene) const override;
	const Vector& color() const;
};