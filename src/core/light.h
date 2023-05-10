#pragma once
#include "vector.h"

class Light {
private:
	Vector m_position;
	Vector m_color;
	float m_intensity;
public:
	__host__ __device__
	Light();
	__host__ __device__
	Light(const Vector &_position, const Vector &color, float _intensity);

	__host__ __device__
	const Vector& position() const;
	__host__ __device__
	const Vector& color() const;
	__host__ __device__
	float intensity() const;
};