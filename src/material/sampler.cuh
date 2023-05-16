#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "../core/vector.h"
#include <cuda_runtime.h>
#include <float.h>

namespace sampler {
	__host__ __device__
	float sample(const Vector &point, unsigned char *dump, float scale) {
		int dimX = 128, dimY = 256, dimZ = 256;
		unsigned char isovalue = 12;
		bool isoRender = false;

		if (dump == nullptr) {
			//printf("Dump not initialised!\n");
			return 0.0f;
		}

		int x = static_cast<int>(point.x * scale) + dimX / 2;
		int y = static_cast<int>(point.y * scale) + dimY / 2;
		int z = static_cast<int>(point.z * scale) + dimZ / 2;

		if (
			x < 0 || x >= dimX
			|| y < 0 || y >= dimY
			|| z < 0 || z >= dimZ
			) {
			//printf("Out of bounds volume access!\n");
			return 0.0f;
		}
		else {
			//printf("Returning volume density =%f\n", dump[x * 256 + y * 256 + z]);
			if (!isoRender)
				return static_cast<float>(dump[x * dimY * dimZ + y * dimZ + z] / 255.0f) * 5.0f;
			else 
				return (dump[x * dimY * dimZ + y * dimZ + z] == isovalue) ? 10.0f : 0.0f;
		}
	}
}