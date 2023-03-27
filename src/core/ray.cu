#include "ray.h"

__host__ __device__ Ray::Ray() :
	origin(),
	direction()
{}
__host__ __device__ Ray::Ray(const Vector &_origin, const Vector &_direction) :
	origin(_origin),
	direction(_direction)
{}

__host__ __device__ Vector Ray::getHit() {
	return origin + hit * direction;
}