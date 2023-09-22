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
	Vector m_position;
	bool m_volume;

protected:
	__host__ __device__
	Object();
	__host__ __device__
	Object(const Vector& _position, bool _volume);
	__host__ __device__
	Object(const Object &other);

public:
	__host__ __device__
	bool isVolume() const;

	__host__ __device__
	virtual float intersect(const Ray& ray) const = 0;
	__host__ __device__
	virtual Vector normal(const Vector &point) const = 0;
	__host__ __device__
	const Vector& position() const;
};