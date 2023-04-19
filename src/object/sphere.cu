#include "sphere.h"

Sphere::Sphere(const Vector &_position, float _radius, const std::shared_ptr<Material>& _material, bool _volume) :
	Object(_material, _position, _volume),
	m_radius(_radius)
{}

float Sphere::intersect(const Ray& ray) const {
	float a = dot(ray.direction(), ray.direction());
	float b = 2 * dot(ray.origin() - this->position(), ray.direction());
	float c = dot(ray.origin(), ray.origin()) + dot(this->position(), this->position()) - 2 * dot(ray.origin(), this->position()) - m_radius * m_radius;
	float discriminant = b * b - 4.0*a*c;

	//if discriminant is non negative we have an intersection
	float t = -1;
	if (discriminant >= 0.0)
	{
		if (discriminant == 0)
		{
			t = -b / (2.0f*a);
		}
		else
		{
			//Calculate both roots
			float d = sqrt(discriminant);
			float t1 = (-b + d) / (2.0f*a);
			float t2 = (-b - d) / (2.0f*a);

			t = t1;
			if (t < 0) {
				t = t2;
			}
			else if (t2 >= 0 && t2 < t){
				t = t2;
			}
		}
	}

	return t;
}

Vector Sphere::normal(const Vector &point) const {
	return point - position();
}
float Sphere::radius() {
	return m_radius;
}