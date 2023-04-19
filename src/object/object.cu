#include "object.h"

Object::Object() : m_material(), m_volume(false) {}
Object::Object(const std::shared_ptr<Material>& _material, const Vector& _position, bool _volume) :
	m_material(_material),
	m_volume(_volume),
	m_position(_position)
{}

Object::Object(const Object &other) :
	m_material(other.m_material),
	m_volume(other.m_volume),
	m_position(other.m_position)
{}

const std::shared_ptr<Material>& Object::material() const{
	return m_material;
};
bool Object::isVolume() const{
	return m_volume;
}
Vector Object::color(Ray& ray, const Scene& scene) const {
	return m_material->shade(ray, *this, scene);
}
const Vector& Object::position() const {
	return m_position;
}