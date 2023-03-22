#pragma once
#include "world.h"

__host__ __device__ void World::cast(Ray &ray, std::vector<Object> &objectList) {
	for (const auto &object : objectList) {
		float currHit = object.intersect(ray);
		if (currHit < ray.hit) ray.hit = currHit;
	}
}