#include "object.h"

__host__ __device__
Object::Object() : m_volume(false) {}
__host__ __device__
Object::Object(const Vector& _position, bool _volume) :
	m_volume(_volume),
	m_position(_position)
{}

__host__ __device__
Object::Object(const Object &other) :
	m_volume(other.m_volume),
	m_position(other.m_position)
{}

__host__ __device__
bool Object::isVolume() const{
	return m_volume;
}
__host__ __device__
const Vector& Object::position() const {
	return m_position;
}