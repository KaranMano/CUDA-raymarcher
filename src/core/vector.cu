#include "vector.h"

__host__ __device__ Vector::Vector() : x(), y(), z() {};
__host__ __device__ Vector::Vector(const float &_x, const float &_y, const float &_z) : x(_x), y(_y), z(_z) {};
__host__ __device__ Vector::Vector(const Vector &v) : x(v.x), y(v.y), z(v.z) {};

__host__ __device__ Vector Vector::operator-() const {
	return {-x, -y, -z};
}
__host__ __device__ Vector& Vector::operator+=(const Vector& rhs) {
	x += rhs.x;
	y += rhs.y;
	z += rhs.z;
	return *this;
}
__host__ __device__ Vector& Vector::operator-=(const Vector& rhs) {
	x -= rhs.x;
	y -= rhs.y;
	z -= rhs.z;
	return *this;
}
__host__ __device__ Vector& Vector::operator/=(const Vector& rhs) {
	x /= rhs.x;
	y /= rhs.y;
	z /= rhs.z;
	return *this;
}
__host__ __device__ Vector& Vector::operator*=(const Vector& rhs) {
	x *= rhs.x;
	y *= rhs.y;
	z *= rhs.z;
	return *this;
}
__host__ __device__ Vector& Vector::operator*=(float scalar) {
	x *= scalar;
	y *= scalar;
	z *= scalar;
	return *this;
}

__host__ __device__ Vector& Vector::operator=(const Vector& rhs){
	x = rhs.x;
	y = rhs.y;
	z = rhs.z;
	return *this;
}
__host__ __device__ Vector Vector::operator+(const Vector& rhs) const {
	Vector _tmp(*this);
	_tmp += rhs;
	return _tmp;
}
__host__ __device__ Vector Vector::operator-(const Vector& rhs) const {
	Vector _tmp(*this);
	_tmp -= rhs;
	return _tmp;
}
__host__ __device__ Vector Vector::operator*(const Vector& rhs) const {
	Vector _tmp(*this);
	_tmp *= rhs;
	return _tmp;
}
__host__ __device__ Vector Vector::operator/(const Vector& rhs) const {
	Vector _tmp(*this);
	_tmp /= rhs;
	return _tmp;
}

__host__ __device__ Vector Vector::operator*(float scalar) const {
	Vector _tmp(*this);
	_tmp *= scalar;
	return _tmp;
}
__host__ __device__ Vector operator*(float scalar, const Vector& rhs) {
	Vector _tmp(rhs);
	_tmp *= scalar;
	return _tmp;
}

__host__ __device__ Vector cross(const Vector& lhs, const Vector &rhs) {
	return {
		lhs.y * rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z,
		lhs.x * rhs.y - lhs.y * rhs.x
	};
}
__host__ __device__ float dot(const Vector& lhs, const Vector &rhs) {
	return (
		lhs.x * rhs.x
		+ lhs.y * rhs.y
		+ lhs.z * rhs.z
	);
}