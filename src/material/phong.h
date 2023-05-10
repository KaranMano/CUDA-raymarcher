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
	__host__ __device__
	Phong();
	__host__ __device__
	Phong(const Vector &_color);
	__host__ __device__
	Phong(const Phong &other);

	__host__ __device__
	Vector shade(Ray& ray, const Object& object, Scene& scene) const override;

	__host__ __device__
	const Vector& color();
	__host__ __device__
	float ambient();
	__host__ __device__
	float diffuse();
	__host__ __device__
	float specular();
	__host__ __device__
	int alpha();
};