#include "ray.h"
#include "vector.h"
#include <cuda_runtime.h>

class Material {
	public:
		__host__ __device__ virtual Vector shade(const Ray& ray) const;
};