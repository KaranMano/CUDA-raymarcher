#pragma once
#include "ray.h"
#include "light.h"
#include "../object/object.h"
#include "camera.h"
#include <memory>

class Scene {
private:
	std::vector<std::shared_ptr<Object>> m_objects;
	std::vector<std::shared_ptr<Light>> m_lights;
	Camera m_camera;
	Vector m_background;
public:
	Scene();

	const std::vector<std::shared_ptr<Object>>& objects() const;
	const std::vector<std::shared_ptr<Light>>& lights() const;
	const Camera& camera() const;
	void cast(Ray& ray) const;
	void add(const std::shared_ptr<Light>& light);
	void add(const std::shared_ptr<Object>& object);
	const Vector& background() const;
};