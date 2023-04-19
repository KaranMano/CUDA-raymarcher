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
	Material();
	Material(const Material &other);

public:
	virtual Vector shade(Ray& ray, const Object& object, const Scene& scene) const = 0;
};