#include "scene.h"

__host__ __device__
Scene::Scene() :
	m_objects(),
	m_lights(),
	m_camera(),
	m_background(0.3, 0.3, 0.5),
	m_numberOfObjects(0),
	m_numberOfLights(0)
{
	for (int i = 0; i < MAX; i++) {
		m_objects[i] = nullptr;
		m_materials[i] = nullptr;
		m_lights[i] = nullptr;
	}
}

__host__ __device__
Scene::Scene(int _width, int _height) :
	m_objects(),
	m_lights(),
	m_camera(_width, _height),
	m_background(0.3, 0.3, 0.5),
	m_numberOfObjects(0),
	m_numberOfLights(0)
{
	for (int i = 0; i < MAX; i++) {
		m_objects[i] = nullptr;
		m_materials[i] = nullptr;
		m_lights[i] = nullptr;
	}
}

__host__ __device__
Object** Scene::objects() {
	return m_objects;
}
__host__ __device__
Light** Scene::lights() {
	return m_lights;
}
__host__ __device__
Material** Scene::materials() {
	return m_materials;
}
__host__ __device__
const Camera& Scene::camera() const {
	return m_camera;
}
__host__ __device__
void Scene::cast(Ray& ray) const {
	int i = 0;
	while (m_objects[i] != nullptr) {
		ray.hit(m_objects[i]->intersect(ray), i);
		i++;
	}
}
__host__ __device__
void Scene::add(Light* light) {
	if (m_numberOfLights < MAX) {
		m_lights[m_numberOfLights] = light;
		m_numberOfLights++;
	}
}
__host__ __device__
void Scene::add(Object* object, Material* material) {
	if (m_numberOfObjects < MAX) {
		m_objects[m_numberOfObjects] = object;
		m_materials[m_numberOfObjects] = material;
		m_numberOfObjects++;
	}
}
__host__ __device__
const Vector& Scene::background() const {
	return m_background;
}