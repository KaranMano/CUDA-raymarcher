#include "vector.h"

Vector::Vector() : x(), y(), z() {};
Vector::Vector(const float &_x, const float &_y, const float &_z) : x(_x), y(_y), z(_z) {};
Vector::Vector(const Vector &v) : x(v.x), y(v.y), z(v.z) {};

std::ostream& operator<<(std::ostream& os, const Vector& v) {
	os << "( " << v.x << ", " << v.y << ", " << v.z << ")";
	return os;
}
Vector Vector::operator-() const {
	return { -x, -y, -z };
}
Vector& Vector::operator+=(const Vector& rhs) {
	x += rhs.x;
	y += rhs.y;
	z += rhs.z;
	return *this;
}
Vector& Vector::operator-=(const Vector& rhs) {
	x -= rhs.x;
	y -= rhs.y;
	z -= rhs.z;
	return *this;
}
Vector& Vector::operator/=(const Vector& rhs) {
	x /= rhs.x;
	y /= rhs.y;
	z /= rhs.z;
	return *this;
}
Vector& Vector::operator*=(const Vector& rhs) {
	x *= rhs.x;
	y *= rhs.y;
	z *= rhs.z;
	return *this;
}
Vector& Vector::operator*=(float scalar) {
	x *= scalar;
	y *= scalar;
	z *= scalar;
	return *this;
}
Vector& Vector::operator/=(float scalar) {
	x /= scalar;
	y /= scalar;
	z /= scalar;
	return *this;
}

Vector& Vector::operator=(const Vector& rhs) {
	x = rhs.x;
	y = rhs.y;
	z = rhs.z;
	return *this;
}
Vector Vector::operator+(const Vector& rhs) const {
	Vector _tmp(*this);
	_tmp += rhs;
	return _tmp;
}
Vector Vector::operator-(const Vector& rhs) const {
	Vector _tmp(*this);
	_tmp -= rhs;
	return _tmp;
}
Vector Vector::operator*(const Vector& rhs) const {
	Vector _tmp(*this);
	_tmp *= rhs;
	return _tmp;
}
Vector Vector::operator/(const Vector& rhs) const {
	Vector _tmp(*this);
	_tmp /= rhs;
	return _tmp;
}

Vector Vector::operator*(float scalar) const {
	Vector _tmp(*this);
	_tmp *= scalar;
	return _tmp;
}
Vector operator*(float scalar, const Vector& rhs) {
	Vector _tmp(rhs);
	_tmp *= scalar;
	return _tmp;
}

Vector cross(const Vector& lhs, const Vector &rhs) {
	return {
		lhs.y * rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z,
		lhs.x * rhs.y - lhs.y * rhs.x
	};
}
float dot(const Vector& lhs, const Vector &rhs) {
	return (
		lhs.x * rhs.x
		+ lhs.y * rhs.y
		+ lhs.z * rhs.z
		);
}
float length(const Vector& v) {
	return std::sqrt(dot(v, v));
}
Vector normalize(const Vector& v) {
	float len = length(v);
	return {
		v.x / len,
		v.y / len,
		v.z / len,
	};
}