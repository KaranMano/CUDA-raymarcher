#include "light.h"

Light::Light() : m_position(), m_color(1.0f, 1.0f, 1.0f), m_intensity(1.0) {}
Light::Light(const Vector &_position, const Vector &_color, float _intensity) : 
	m_position(_position), 
	m_color(_color),
	m_intensity(_intensity)
{}

const Vector& Light::position() const{
	return m_position;
}

const Vector& Light::color() const{
	return m_color;
}

float Light::intensity() const {
	return m_intensity;
}