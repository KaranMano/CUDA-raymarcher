#pragma once
#include "vector.h"
#include <vector>
#include <cuda_runtime.h>

class Ray {
	public:
		Vector origin, direction;
		float hit;
		
		__host__ __device__ Ray();
		__host__ __device__ Ray(const Vector &_origin, const Vector &_direction);

		__host__ __device__ Vector getHit();
};