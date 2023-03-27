#pragma once
#include "ray.h"
#include "object.h"
#include "list.h"
#include <thrust/pair.h>
#include <vector>
#include <cuda_runtime.h>

class World {
	public:
	List objectList;

	__host__ __device__ void cast(Ray &ray, List &objectList);
};