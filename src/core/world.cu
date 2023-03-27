#include "world.h"

__host__ __device__ void World::cast(Ray &ray, List &objectList) {
	Node *curr = objectList.head;
	while (curr != nullptr) {
		thrust::pair<bool, float> currHit = curr->obj.intersect(ray);
		if (currHit.first && currHit.second < ray.hit) 
			ray.hit = currHit.second;
		curr = curr->next;
	}
}