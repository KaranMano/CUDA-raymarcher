#pragma once
#include "vector.h"
#include "ray.h"
#include <cuda_runtime.h>

class Object {
	public:
		__host__ __device__ virtual float intersect(const Ray& ray) const;
};