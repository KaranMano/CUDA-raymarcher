#pragma once
#include "vector.h"
#include "ray.h"
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include <iostream>

class Camera {
private:
	int m_imageHeight, m_imageWidth;
	float m_fov;
	Vector m_position, m_view, m_up; // up and view should be perpendicular
	const float IMAGE_PLANE_DISTANCE = 1.0f;
public:
	__host__ __device__
		Camera();
	__host__ __device__
		Camera(
			int _imageWidth,
			int _imageHeight
		);
	__host__ __device__
		Camera(
			const Vector &_position,
			const Vector &_view,
			const Vector &_up,
			int _imageWidth,
			int _imageHeight,
			float _fov
		);

	__host__ __device__
		Ray ray(int row, int col) const;
	__host__ __device__
		int height() const;
	__host__ __device__
		int width() const;
	__host__ __device__
		float fov() const;
	__host__ __device__
		const Vector& position() const;
	__host__ __device__
		const Vector& view() const;
	__host__ __device__
		const Vector& up() const;
};