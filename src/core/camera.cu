#include "camera.h"
		
__host__ __device__ Camera::Camera() : 
	position(0.0f, 0.0f, 0.0f),
	view(0.0f, 0.0f, -1.0f), 
	up(0.0f, 1.0f, 0.0f),
	near(0.1f)
{}

__host__ __device__ Camera::Camera(
		const Vector &_position, 
		const Vector &_view, 
		const Vector &_up, 
		float _near,
		int _imageCols,
		int _imageRows
) : 
	position(_position),
	view(_view), 
	up(_up),
	near(_near),
	imageCols(_imageCols),
	imageRows(_imageRows)
{}

__host__ __device__ Ray Camera::getRay(int row, int col) {
	Vector rayDirCam(-(col - (imageRows / 2.0f)), (row - (imageRows / 2.0f)), near);
	Vector xBasis = cross(up, view);
	Vector rayDir = xBasis * rayDirCam.x + up * rayDirCam.y - view * rayDirCam.z;
	return { position, rayDir };
}
	
