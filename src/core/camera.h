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

	  Camera();
	  Camera(
		const Vector &_position,
		const Vector &_view,
		const Vector &_up,
		int _imageWidth,
		int _imageHeight,
		float _fov
	);

	 Ray ray(int row, int col) const;
	 int height() const;
	 int width() const;
	 float fov() const;
	 const Vector& position() const;
	 const Vector& view() const;
	 const Vector& up() const;
};