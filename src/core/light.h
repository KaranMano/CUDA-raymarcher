#pragma once
#include "vector.h"

class Light {
private:
	Vector m_position;
	Vector m_color;
	float m_intensity;
public:

	Light();
	Light(const Vector &_position, const Vector &color, float _intensity);

	const Vector& position() const;
	const Vector& color() const;
	float intensity() const;
};