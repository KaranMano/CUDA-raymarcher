#include "phong.h"

Phong::Phong() :
	Material(),
	m_color(1.0f, 1.0f, 1.0f),
	m_ambient(0.5f),
	m_diffuse(0.6f),
	m_specular(0.1f),
	m_alpha(2)
{}
Phong::Phong(const Vector &_color) :
	Material(),
	m_color(_color),
	m_ambient(0.5f),
	m_diffuse(0.6f),
	m_specular(0.1f),
	m_alpha(2)
{}
Phong::Phong(const Phong &other) :
	Material(other),
	m_color(other.m_color),
	m_ambient(other.m_ambient),
	m_diffuse(other.m_diffuse),
	m_specular(other.m_specular),
	m_alpha(other.m_alpha)
{}

Vector Phong::shade(Ray& ray, const Object& object, const Scene& scene) const {

	const Vector intersection = ray.hit();
	const Vector view = normalize(scene.camera().position() - intersection);

	Vector I(0.0f, 0.0f, 0.0f);
	Vector diffuseIntensity(0.0f, 0.0f, 0.0f);
	Vector specularIntensity(0.0f, 0.0f, 0.0f);
	Vector ambientIntensity(1.0f, 1.0f, 1.0f);

	for (const auto& light : scene.lights()) {
		Vector normal = object.normal(intersection);
		Vector lightDir = light->position() - intersection;
		float lightDistanceSquared = dot(lightDir, lightDir);
		lightDir = normalize(lightDir);
		Vector halfway = normalize(lightDir + view);

		diffuseIntensity += m_diffuse * light->color() * std::max(dot(lightDir, normal), 0.0f);
		specularIntensity = m_specular * light->color() * std::max(std::pow(dot(halfway, normal), m_alpha), 0.0f);
		I += (diffuseIntensity + specularIntensity) * std::clamp(1.0f / (0.02f * lightDistanceSquared), 0.0f, 1.0f);
	}
	ambientIntensity = m_ambient * Vector(1.0f, 1.0f, 1.0f);
	I += ambientIntensity;


	return I * m_color;
}

const Vector& Phong::color() {
	return m_color;
}
float Phong::ambient() {
	return m_ambient;
}
float Phong::diffuse() {
	return m_diffuse;
}
float Phong::specular() {
	return m_specular;
}
int Phong::alpha() {
	return m_alpha;
}