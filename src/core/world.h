#pragma once
#include "ray.h"
#include "object.h"
#include <vector>
#include <cuda_runtime.h>

class World {
	public:
		std::vector<Object> objectList;

		__host__ __device__ void cast(Ray &ray, std::vector<Object> &objectList);
};