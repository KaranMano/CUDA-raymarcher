#include "Volume.h"
#include <random>
#include <functional>

Volume::Volume() :
	Material(),
	m_color(1.0f, 1.0f, 1.0f)
{}
Volume::Volume(const Vector &_color) :
	Material(),
	m_color(_color)
{}
Volume::Volume(const Volume &other) :
	Material(other)
{}

float Volume::density(const Vector& point) const {
	return (float)std::rand() / RAND_MAX;
}

//! move g to volume property
float Volume::phase(float g, float cos ) const {
	const float PI = 3.14f;
	return (1.0f - g*g) / (4*PI * std::pow(1 + g*g - 2*g*cos, 3.0f / 2.0f));
}


//! storing opacity and color in ray
//! there may be an error in the light scattering step
//! maybe move light intensity into colour?
//! maybe adaptive sampling and adaptive step size
//! randomized sampling to remove banding
//! camera within volume?
//! multiple light factor multiplication optimization
//!
Vector Volume::shade(Ray& ray, const Object& object, const Scene& scene) const {
	const float PI = 3.14f;
	float transmittance = 1.0f;
	float absorptionCoeff = 0.1f;
	float scatteringCoeff = 0.1f;
	float g = 0.4;
	float dropoutFactor = 5.0f;
	float stepSize = 2 * std::tanf(PI / 180 * scene.camera().fov() / (2 * scene.camera().width())) * ray.param();// 0.1f;
	Vector lightColor(0.0f, 0.0f, 0.0f);

	bool inVolume = true;
	ray.origin(ray.hit() + 0.001f * ray.direction()); // moving origin into the volume from the surface
	while (inVolume) {
		float density = this->density(ray.origin());
		transmittance *= std::exp(-stepSize * density * (absorptionCoeff + scatteringCoeff));

		for (const auto& light : scene.lights()) {
			Ray lightRay(ray.origin(), normalize(light->position() - ray.origin()));
			float distance = object.intersect(lightRay);
			lightColor +=
				transmittance
				* this->phase(g, dot(lightRay.direction(), -ray.direction())) // direction from object to eye for both rays
				* density
				* scatteringCoeff
				* std::exp(-distance * (density * absorptionCoeff + scatteringCoeff)) 
				* light->intensity()
				* stepSize 
				* light->color();
		}

		ray.origin(ray.origin() + (stepSize * ray.direction()));
		ray.reset();
		inVolume = object.intersect(ray) >= 0;
		if (transmittance < 1e-3) {
			if (std::rand() > 1 / dropoutFactor) {
				break;
			}
			else {
				transmittance *= dropoutFactor;
			}
		}
	}
	scene.cast(ray);
	Vector backgroundColor;
	if (ray.intersected() != nullptr)
		backgroundColor = ray.intersected()->color(ray, scene);
	else
		backgroundColor = scene.background();
	return (transmittance * backgroundColor) + lightColor;
}

const Vector& Volume::color() const {
	return m_color;
}