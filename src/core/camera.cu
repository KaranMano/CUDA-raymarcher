#include "camera.h"		
Camera::Camera() :
	m_position(0.0f, 0.0f, 0.0f),
	m_view(0.0f, 0.0f, -1.0f),
	m_up(0.0f, 1.0f, 0.0f),
	m_imageWidth(1920),
	m_imageHeight(1080),
	m_fov(90.0f)
{}

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

Ray Camera::ray(int row, int col) const {
	const float PI = 3.14f;
	float rowRandom = std::rand() / static_cast<float>(RAND_MAX);
	float colRandom = std::rand() / static_cast<float>(RAND_MAX);
	float pixelRow = 0.5f / m_imageHeight * std::tan(m_fov / 2 * PI / 180);
	float pixelCol = 0.5f / m_imageWidth * std::tan(m_fov / 2 * PI / 180) * ((float)m_imageWidth / m_imageHeight);

	float remappedRow =
		(1.0f - 2.0f * ((row + 0.5f) / m_imageHeight))
		* std::tan(m_fov / 2 * PI / 180)
		+
		(2 * (pixelRow * rowRandom) - pixelRow);

	float remappedCol = (2.0f * ((col + 0.5f) / m_imageWidth) - 1.0f)
		* std::tan(m_fov / 2 * PI / 180)
		* ((float)m_imageWidth / m_imageHeight)
		+
		(2 * (pixelCol * colRandom) - pixelCol);

	Vector pixelWorldSpace = remappedCol * cross(m_up, -m_view) + remappedRow * m_up + (-IMAGE_PLANE_DISTANCE) * (-m_view);
	return { pixelWorldSpace, normalize(pixelWorldSpace - m_position) };
}

int Camera::height() const {
	return m_imageHeight;
}
int Camera::width() const {
	return m_imageWidth;
}
float Camera::fov() const {
	return m_fov;
}
const Vector& Camera::position() const {
	return m_position;
}
const Vector& Camera::view() const {
	return m_view;
}
const Vector& Camera::up() const {
	return m_up;
}