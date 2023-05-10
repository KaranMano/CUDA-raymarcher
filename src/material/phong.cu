#include "phong.h"

using namespace std;

__host__ __device__
Phong::Phong() :
	Material(),
	m_color(1.0f, 1.0f, 1.0f),
	m_ambient(0.5f),
	m_diffuse(0.6f),
	m_specular(0.1f),
	m_alpha(2)
{}
__host__ __device__
Phong::Phong(const Vector &_color) :
	Material(),
	m_color(_color),
	m_ambient(0.5f),
	m_diffuse(0.6f),
	m_specular(0.1f),
	m_alpha(2)
{}
__host__ __device__
Phong::Phong(const Phong &other) :
	Material(other),
	m_color(other.m_color),
	m_ambient(other.m_ambient),
	m_diffuse(other.m_diffuse),
	m_specular(other.m_specular),
	m_alpha(other.m_alpha)
{}

__host__ __device__
Vector Phong::shade(Ray& ray, const Object& object, Scene& scene) const {

	const Vector intersection = ray.hit();
	const Vector view = normalize(scene.camera().position() - intersection);

	Vector I(0.0f, 0.0f, 0.0f);
	Vector diffuseIntensity(0.0f, 0.0f, 0.0f);
	Vector specularIntensity(0.0f, 0.0f, 0.0f);
	Vector ambientIntensity(1.0f, 1.0f, 1.0f);

	int i = 0;
	while (scene.lights()[i] != nullptr) {
		Light* light = scene.lights()[i];
		Vector normal = object.normal(intersection);
		Vector lightDir = light->position() - intersection;
		float lightDistanceSquared = dot(lightDir, lightDir);
		lightDir = normalize(lightDir);
		Vector halfway = normalize(lightDir + view);

		diffuseIntensity += m_diffuse * light->color() * max(dot(lightDir, normal), 0.0f);
		specularIntensity = m_specular * light->color() * max(pow(dot(halfway, normal), m_alpha), 0.0f);
		I += (diffuseIntensity + specularIntensity) * clamp(1.0f / (0.02f * lightDistanceSquared), 0.0f, 1.0f);
		i++;
	}
	ambientIntensity = m_ambient * Vector(1.0f, 1.0f, 1.0f);
	I += ambientIntensity;

	return I * m_color;
}

__host__ __device__
const Vector& Phong::color() {
	return m_color;
}
__host__ __device__
float Phong::ambient() {
	return m_ambient;
}
__host__ __device__
float Phong::diffuse() {
	return m_diffuse;
}
__host__ __device__
float Phong::specular() {
	return m_specular;
}
__host__ __device__
int Phong::alpha() {
	return m_alpha;
}