#pragma once
class Scene;
#include "../core/ray.h"
#include "../core/vector.h"
#include "../core/scene.h"
#include "../object/object.h"
#include <algorithm>
#include <cuda_runtime.h>

class Material {
protected:
	__host__ __device__
	Material();
	__host__ __device__
	Material(const Material &other);

public:
	__host__ __device__
	virtual Vector shade(Ray& ray, const Object& object, Scene& scene) const = 0;
};