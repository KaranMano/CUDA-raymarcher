#pragma once
class Scene;
class Material;
#include "../core/ray.h"
#include "../core/vector.h"
#include "../core/scene.h"
#include "../material/material.h"
#include <thrust/pair.h>
#include <cuda_runtime.h>
#include <memory>

class Object {
private:
	std::shared_ptr<Material> m_material;
	Vector m_position;
	bool m_volume;

protected:
	Object();
	Object(const std::shared_ptr<Material>& _mat, const Vector& _position, bool _volume);
	Object(const Object &other);

public:
	const std::shared_ptr<Material>& material() const;
	bool isVolume() const;

	virtual float intersect(const Ray& ray) const = 0;
	virtual Vector normal(const Vector &point) const = 0;
	Vector color(Ray& ray, const Scene& scene) const;
	const Vector& position() const;
};