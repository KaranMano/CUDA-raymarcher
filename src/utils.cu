#include "utils.h"

__host__ __device__
unsigned char clamp(int value) {
	if (value > 255)
		return 255;
	else if (value < 0)
		return 0;
	else
		return value;
}

__host__ __device__
Vector renderPixel(int row, int col, Scene& scene) {
	Ray ray = scene.camera().ray(row, col);

	scene.cast(ray);
	if (ray.intersected() == INT_MAX)
		return scene.background();
	else {
		return scene.materials()[ray.intersected()]->shade(ray, *(scene.objects()[ray.intersected()]), scene);
	}
}

__global__
void renderKernel(unsigned char* image, Scene* scene) {
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;

	Vector color = renderPixel(row, col, *scene);
	image[(col + scene->camera().width() * row) * 3 + 0] = clamp(color.x * 255);
	image[(col + scene->camera().width() * row) * 3 + 1] = clamp(color.y * 255);
	image[(col + scene->camera().width() * row) * 3 + 2] = clamp(color.z * 255);
}

__global__
void launchKernel(unsigned char* image, Scene* scene, int blockSize) {
	renderKernel << <dim3((scene->camera().height() + blockSize - 1) / blockSize, (scene->camera().width() + blockSize - 1) / blockSize), dim3(blockSize, blockSize) >> > (image, scene);
}

__global__
void setupKernel(Scene *scene, unsigned char* dump) {
	Volume* volume = new Volume(dump);

	scene->add((Object*)new Sphere({ 0.0f, 0.0f, -10.0f }, 6.0f, true), (Material*)volume);
	scene->add(new Light({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 10.0));
}

__global__
void cleanupKernel(Scene* scene) {
	int i = 0;
	while (scene->objects()[i] != nullptr) {
		delete scene->objects()[i];
		scene->objects()[i] = nullptr;
		if (scene->materials()[i] != nullptr) // can be shared
		{
			delete scene->materials()[i];
			scene->materials()[i] = nullptr;
		}
		i++;
	}
	i = 0;
	while (scene->lights()[i] != nullptr) {
		delete scene->lights()[i];
		scene->lights()[i] = nullptr;
		i++;
	}
}

void renderScene(unsigned char *image, Scene& scene, int offset, int band) {
	for (int row = 0; row < scene.camera().height(); row++)
		for (int col = offset; col < offset + band; col++)
		{
			Vector color = renderPixel(row, col, scene);
			image[(col + scene.camera().width() * row) * 3 + 0] = clamp(color.x * 255);
			image[(col + scene.camera().width() * row) * 3 + 1] = clamp(color.y * 255);
			image[(col + scene.camera().width() * row) * 3 + 2] = clamp(color.z * 255);
		}
}

void cpuKernel(unsigned char* image, Scene &scene, int steps){
	int band = scene.camera().width() / steps;
	std::vector<std::future<void>> jobs;
	for (int step = 0; step < steps; step++) {
		jobs.emplace_back(std::async(std::launch::async, renderScene, image, scene, step * band, band));
	}
	for (auto &job : jobs) {
		job.get();
	}
}

void checkCudaErrors(cudaError_t err) {
	if (err != cudaSuccess) {
		std::cout << "CUDA error : " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
}