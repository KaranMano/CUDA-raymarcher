#pragma once
#include "../material/material.h"

class Phong : public Material {
private:
	Vector m_color;
	float m_ambient;
	float m_diffuse;
	float m_specular;
	int m_alpha;

public:
	Phong();
	Phong(const Vector &_color);
	Phong(const Phong &other);

	Vector shade(Ray& ray, const Object& object, const Scene& scene) const override;

	const Vector& color();
	float ambient();
	float diffuse();
	float specular();
	int alpha();
};