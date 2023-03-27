#include "object.h"

__host__ __device__ thrust::pair<bool, float> Object::intersect(const Ray& ray) const {
	thrust::pair<bool, float> hit;
	hit.first = false;
	hit.second = -1;
	return hit;
}