#include "Volume.h"
#include <random>
#include <functional>
#include "cuda_noise.cuh"

using namespace std;

__host__ __device__
Volume::Volume() :
	Material(),
	m_color(1.0f, 1.0f, 1.0f)
{}
__host__ __device__
Volume::Volume(const Vector &_color) :
	Material(),
	m_color(_color)
{}
__host__ __device__
Volume::Volume(const Volume &other) :
	Material(other)
{}

__host__ __device__
float Volume::density(const Vector& point) const {
#ifdef __CUDA_ARCH__
	float3 pos = make_float3(point.x, point.y, point.z);
	return max(cudaNoise::simplexNoise(pos, 1.0f, 100), 0.0f);
#else
	return max(cudaNoise::cpuNoise::simplexNoise(point, 1.0f, 100), 0.0f);;
#endif
}

//! move g to volume property
__host__ __device__
float Volume::phase(float g, float cos) const {
	const float PI = 3.14f;
	return (1.0f - g * g) / (4 * PI * pow(1 + g * g - 2 * g*cos, 3.0f / 2.0f));
}


//! storing opacity and color in ray
//! there may be an error in the light scattering step
//! maybe move light intensity into colour?
//! maybe adaptive sampling and adaptive step size
//! randomized sampling to remove banding
//! camera within volume?
//! multiple light factor multiplication optimization
//!
__host__ __device__
Vector Volume::shade(Ray& ray, const Object& object, Scene& scene) const {
	const float PI = 3.14f;
	float transmittance = 1.0f;
	float absorptionCoeff = 0.2f;
	float scatteringCoeff = 0.2f;
	float g = 0.24;
	float dropoutFactor = 5.0f;
	float stepSize = 2 * tanf(PI / 180 * scene.camera().fov() / (2 * scene.camera().width())) * ray.param();// 0.1f;
	Vector lightColor(0.0f, 0.0f, 0.0f);

	bool inVolume = true;
	ray.origin(ray.hit() + 0.001f * ray.direction()); // moving origin into the volume from the surface
	while (inVolume) {
		float density = this->density(ray.origin());
		transmittance *= exp(-stepSize * density * (absorptionCoeff + scatteringCoeff));

		int i = 0;
		while (scene.lights()[i] != nullptr) {
			Light* light = scene.lights()[i];
			Ray lightRay(ray.origin(), normalize(light->position() - ray.origin()));
			float distance = object.intersect(lightRay);

			lightColor +=
				transmittance
				* this->phase(g, dot(lightRay.direction(), -ray.direction())) // direction from object to eye for both rays
				* density
				* scatteringCoeff
				* exp(-distance * (density * absorptionCoeff + scatteringCoeff))
				* light->intensity()
				* stepSize
				* light->color();
			i++;
		}

		ray.origin(ray.origin() + (stepSize * ray.direction()));
		ray.reset();
		inVolume = object.intersect(ray) >= 0;
		if (transmittance < 1e-3) {
			if (rand() > 1 / dropoutFactor) {
				break;
			}
			else {
				transmittance *= dropoutFactor;
			}
		}
	}
	scene.cast(ray);
	Vector backgroundColor;
	if (ray.intersected() == INT_MAX)
		backgroundColor = scene.background();
	else
		scene.materials()[ray.intersected()]->shade(ray, *(scene.objects()[ray.intersected()]), scene);
	if (lightColor.x < 0 || lightColor.y < 0 || lightColor.z < 0)
		printf("lightcolor negative!\n");
	return (transmittance * backgroundColor) + lightColor;
}

__host__ __device__
const Vector& Volume::color() const {
	return m_color;
}