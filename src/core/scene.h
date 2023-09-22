#pragma once
#include "ray.h"
#include "light.h"
#include "../object/object.h"
#include "camera.h"
#include <memory>

#define MAX 1024

class Scene {
private:
	int m_numberOfObjects, m_numberOfLights;
	Object* m_objects[MAX];
	Material* m_materials[MAX];
	Light* m_lights[MAX];
	Camera m_camera;
	Vector m_background;
public:
	__host__ __device__
		Scene();
	__host__ __device__
		Scene(int _width, int _height);

	__host__ __device__
		Object** objects();
	__host__ __device__
		Light** lights();
	__host__ __device__
		Material** materials();
	__host__ __device__
		const Camera& camera() const;
	__host__ __device__
		void cast(Ray& ray) const;
	__host__ __device__
		void add(Light* light);
	__host__ __device__
		void add(Object* object, Material* material);
	__host__ __device__
		const Vector& background() const;
};