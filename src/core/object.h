#pragma once
#include "vector.h"
#include "ray.h"
#include <thrust/pair.h>
#include <cuda_runtime.h>

class Object {
	public:
		__host__ __device__ virtual thrust::pair<bool, float> intersect(const Ray& ray) const;
};