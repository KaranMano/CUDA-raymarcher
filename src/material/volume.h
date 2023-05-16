#pragma once
#include "../material/material.h"
#include <cmath>
#include <random>
class Volume : public Material {
private:
	Vector m_color;
	int m_p[512];
	unsigned char *m_dump;
public:
	__host__ __device__
		Volume();
	__host__ __device__
		Volume(unsigned char* _dump);
	__host__ __device__
		Volume(const Vector &_color);
	__host__ __device__
		Volume(const Volume  &other);

	__host__ __device__
		float density(const Vector& point) const;
	__host__ __device__
		float phase(float g, float cos) const;
	__host__ __device__
		float noise(const Vector &v) const;
	__host__ __device__
		Vector shade(Ray& ray, const Object& object, Scene& scene) const override;
	__host__ __device__
		const Vector& color() const;
};