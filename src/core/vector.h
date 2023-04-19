#pragma once
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>

class Vector {
public:
	float x, y, z;
	Vector();
	Vector(const float &_x, const float &_y, const float &_z);
	Vector(const Vector &v);

	friend std::ostream& operator<<(std::ostream& os, const Vector& v);
	Vector operator-() const;
	Vector& operator+=(const Vector& rhs);
	Vector& operator-=(const Vector& rhs);
	Vector& operator/=(const Vector& rhs);
	Vector& operator*=(const Vector& rhs);
	Vector& operator/=(float scalar);
	Vector& operator*=(float scalar);

	Vector& operator=(const Vector& rhs);
	Vector operator+(const Vector& rhs) const;
	Vector operator-(const Vector& rhs) const;
	Vector operator*(const Vector& rhs) const;
	Vector operator/(const Vector& rhs) const;

	Vector operator*(float scalar) const;
	friend Vector operator*(float scalar, const Vector& rhs);
};

Vector cross(const Vector& lhs, const Vector &rhs);
float dot(const Vector& lhs, const Vector &rhs);
Vector normalize(const Vector& v);
float length(const Vector& v);