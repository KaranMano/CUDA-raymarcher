#include "scene.h"

Scene::Scene() : 
	m_objects(), 
	m_lights(),
	m_camera(),
	m_background(1.0, 0.0, 1.0)
{}

const std::vector<std::shared_ptr<Object>>& Scene::objects() const {
	return m_objects;
}
const std::vector<std::shared_ptr<Light>>& Scene::lights() const {
	return m_lights;
}
const Camera& Scene::camera() const {
	return m_camera;
}
void Scene::cast(Ray& ray) const {
	for (auto& object : m_objects) {
		ray.hit(object->intersect(ray), object);
	}
}
void Scene::add(const std::shared_ptr<Light>& light) {
	m_lights.push_back(light);
}
void Scene::add(const std::shared_ptr<Object>& object) {
	m_objects.push_back(object);
}

const Vector& Scene::background() const{
	return m_background;
}