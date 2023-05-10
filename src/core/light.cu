#include "light.h"

__host__ __device__
Light::Light() : m_position(), m_color(1.0f, 1.0f, 1.0f), m_intensity(1.0) {}
__host__ __device__
Light::Light(const Vector &_position, const Vector &_color, float _intensity) : 
	m_position(_position), 
	m_color(_color),
	m_intensity(_intensity)
{}

__host__ __device__
const Vector& Light::position() const{
	return m_position;
}
__host__ __device__
const Vector& Light::color() const{
	return m_color;
}
__host__ __device__
float Light::intensity() const {
	return m_intensity;
}