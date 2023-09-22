#include "ray.h"

__host__ __device__
Ray::Ray() :
	m_origin(),
	m_direction(),
	m_hit(FLT_MAX),
	m_interected(INT_MAX)
{}
__host__ __device__
Ray::Ray(const Vector &_origin, const Vector &_direction) :
	m_origin(_origin),
	m_direction(_direction),
	m_hit(FLT_MAX),
	m_interected(INT_MAX)
{}
__host__ __device__
float Ray::param() const {
	return m_hit;
}
__host__ __device__
Vector Ray::hit() const {
	return m_origin + m_hit * m_direction;
}
__host__ __device__
void Ray::hit(float _hit, int _intersected) {
	if (_hit >= 0 && _hit < m_hit) {
		m_hit = _hit;
		m_interected = _intersected;
	}
}
__host__ __device__
void Ray::reset() {
	m_hit = FLT_MAX;
	m_interected = INT_MAX;
}
__host__ __device__
const Vector& Ray::origin() const {
	return m_origin;
}
__host__ __device__
void Ray::origin(const Vector& v) {
	m_origin = v;
}
__host__ __device__
const Vector& Ray::direction() const {
	return m_direction;
}
__host__ __device__
int Ray::intersected() const {
	return m_interected;
}