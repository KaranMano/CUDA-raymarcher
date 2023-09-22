#include "camera.h"

__host__ __device__
Camera::Camera() :
	m_position(0.0f, 0.0f, 0.0f),
	m_view(0.0f, 0.0f, -1.0f),
	m_up(0.0f, 1.0f, 0.0f),
	m_imageWidth(1920),
	m_imageHeight(1080),
	m_fov(90.0f)
{}

__host__ __device__
Camera::Camera(
	int _imageWidth,
	int _imageHeight
) :
	m_position(0.0f, 0.0f, 0.0f),
	m_view(0.0f, 0.0f, -1.0f),
	m_up(0.0f, 1.0f, 0.0f),
	m_imageWidth(_imageWidth),
	m_imageHeight(_imageHeight),
	m_fov(90.0f)
{}

__host__ __device__
Camera::Camera(
	const Vector &_position,
	const Vector &_view,
	const Vector &_up,
	int _imageWidth,
	int _imageHeight,
	float _fov
) :
	m_position(_position),
	m_view(_view),
	m_up(_up),
	m_imageWidth(_imageWidth),
	m_imageHeight(_imageHeight),
	m_fov(_fov)
{}

__host__ __device__
Ray Camera::ray(int row, int col) const {
	const float PI = 3.14f;

	float remappedRow =
		(1.0f - 2.0f * ((row + 0.5f) / m_imageHeight))
		* tan(m_fov / 2 * PI / 180);

	float remappedCol = (2.0f * ((col + 0.5f) / m_imageWidth) - 1.0f)
		* tan(m_fov / 2 * PI / 180)
		* ((float)m_imageWidth / m_imageHeight);

	Vector pixelWorldSpace = remappedCol * cross(m_up, -m_view) + remappedRow * m_up + (-IMAGE_PLANE_DISTANCE) * (-m_view);
	return { pixelWorldSpace, normalize(pixelWorldSpace - m_position) };
}

__host__ __device__
int Camera::height() const {
	return m_imageHeight;
}
__host__ __device__
int Camera::width() const {
	return m_imageWidth;
}
__host__ __device__
float Camera::fov() const {
	return m_fov;
}
__host__ __device__
const Vector& Camera::position() const {
	return m_position;
}
__host__ __device__
const Vector& Camera::view() const {
	return m_view;
}
__host__ __device__
const Vector& Camera::up() const {
	return m_up;
}