#pragma once
#include "vector.h"
#include "ray.h"
#include <cuda_runtime.h>

class Camera {
	public:
		Vector position, view, up; // up and view should be perpendicular
		int imageRows, imageCols;
		float near;
		
		__host__ __device__ Camera();
		__host__ __device__ Camera(
				const Vector &_position, 
				const Vector &_view, 
				const Vector &_up, 
				float _near,
				int _imageCols,
				int _imageRows
		);

		__host__ __device__ Ray getRay(int row, int col);
};