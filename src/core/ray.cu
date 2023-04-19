#include "ray.h"

Ray::Ray() :
	m_origin(),
	m_direction(),
	m_hit(FLT_MAX)
{}
Ray::Ray(const Vector &_origin, const Vector &_direction) :
	m_origin(_origin),
	m_direction(_direction),
	m_hit(FLT_MAX)
{}

float Ray::param() const {
	return m_hit;
}
Vector Ray::hit() const {
	return m_origin + m_hit * m_direction;
}
void Ray::hit(float _hit, const std::shared_ptr<Object>& _intersected) {
	if (_hit >= 0 && _hit < m_hit) {
		m_hit = _hit;
		m_interected = _intersected;
	}
}
void Ray::reset() {
	m_hit = FLT_MAX;
	m_interected.reset();
}
const Vector& Ray::origin() const {
	return m_origin;
}
void Ray::origin(const Vector& v) {
	m_origin = v;
}
const Vector& Ray::direction() const {
	return m_direction;
}
const std::shared_ptr<Object>& Ray::intersected() const {
	return m_interected;
}