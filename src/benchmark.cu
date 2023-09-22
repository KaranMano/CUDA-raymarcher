#include "utils.h"
#include "timer.cuh"
#include <string>
#include <iostream>
#include <fstream>
#include <stb_image_write.h>

int main() {
	std::vector<int> imageSizes = {32, 64, 128 , 256, 512, 1080};
	std::vector<int> numberOfCores = { 2, 4, 6, 8, 10 };
	std::vector<int> blockSizes = { 4, 8, 16, 32 };

	std::vector<std::vector<cpu::timer>> cpuTimers(numberOfCores.size(), std::vector <cpu::timer>(imageSizes.size()));
	std::vector<std::vector<gpu::timer>> gpuTimers(blockSizes.size(), std::vector<gpu::timer>(imageSizes.size()));

	for (int j = 0; j < imageSizes.size(); j++) {
		auto imageSize = imageSizes[j];
		std::cout << "image size = " << imageSize << " \n";

		Scene scene(imageSize, imageSize), *d_scene;
		unsigned char* image = (unsigned char *)malloc(scene.camera().width() * scene.camera().height() * CHANNELS * sizeof(unsigned char));
		unsigned char* d_image;

		std::cout << "gpu scene setup\n";
		cudaMalloc(&d_scene, sizeof(scene));
		cudaMemcpy(d_scene, &scene, sizeof(scene), cudaMemcpyHostToDevice);
		cudaMalloc(&d_image, scene.camera().height()* scene.camera().width() * CHANNELS);
		setupKernel << <1, 1 >> > (d_scene);
		checkCudaErrors(cudaDeviceSynchronize());

		std::cout << "cpu scene setup\n";
		scene.add((Object*)new Sphere({ 0.0f, 0.0f, -10.0f }, 6.0f, true), (Material*)new Volume({ 1.0f, 1.0f, 1.0f }));
		scene.add(new Light({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 10.0));

		for (int i = 0; i < numberOfCores.size(); i++) {
			std::string label = "cpu image-size=" + std::to_string(imageSize) + " cores=" + std::to_string(numberOfCores[i]);
			std::cout << label << "\n";
			cpuTimers[i][j].label(label);
			cpuTimers[i][j].start();
			cpuKernel(image, scene, numberOfCores[i]);
			std::cout << "time taken = " << cpuTimers[i][j].end() << "\n";

			stbi_write_png((label + ".png").c_str(), scene.camera().width(), scene.camera().height(), CHANNELS, image, scene.camera().width() * CHANNELS);
		}

		for (int i = 0; i < blockSizes.size(); i++) {
			std::string label = "gpu image-size=" + std::to_string(imageSize) + " blockSize=" + std::to_string(blockSizes[i]);
			std::cout << label << "\n";
			gpuTimers[i][j].label(label);
			gpuTimers[i][j].start();
			launchKernel << <1, 1 >> > (d_image, d_scene, blockSizes[i]);
			std::cout << "time taken = " << gpuTimers[i][j].end() << "\n";
			checkCudaErrors(cudaDeviceSynchronize());
			cudaMemcpy(image, d_image, scene.camera().height()* scene.camera().width() * CHANNELS, cudaMemcpyDeviceToHost);

			stbi_write_png((label + ".png").c_str(), scene.camera().width(), scene.camera().height(), CHANNELS, image, scene.camera().width() * CHANNELS);
		}

		std::cout << "gpu scene cleanup\n";
		cleanupKernel << <1, 1 >> > (d_scene);
		checkCudaErrors(cudaDeviceSynchronize());
		cudaFree(d_image);
		cudaFree(d_scene);

		std::cout << "cpu scene cleanup\n";
		free(image);
		int i = 0;
		while (scene.objects()[i] != nullptr) {
			delete scene.objects()[i];
			scene.objects()[i] = nullptr;
			if (scene.materials()[i] != nullptr) // can be shared
			{
				delete scene.materials()[i];
				scene.materials()[i] = nullptr;
			}
			i++;
		}
		i = 0;
		while (scene.lights()[i] != nullptr) {
			delete scene.lights()[i];
			scene.lights()[i] = nullptr;
			i++;
		}
	}

	std::ofstream timings("./timings.json");
	timings << "{\"GPU\":{\n";
	for (int i = 0; i < imageSizes.size(); i++) {
		timings << "\"" << imageSizes[i] << "\": {\n";
		for (int j = 0; j < blockSizes.size(); j++) {
			timings << "\"" << blockSizes[j] << "\":" << gpuTimers[j][i].get();
			if (j != blockSizes.size() - 1)
				timings << ",\n";
		}
		timings << "\n}";
		if (i != imageSizes.size() - 1)
			timings << ",\n";
	}
	timings << "\n},\n";
	timings << "\"CPU\":{\n";
	for (int i = 0; i < imageSizes.size(); i++) {
		timings << "\"" << imageSizes[i] << "\": {\n";
		for (int j = 0; j < numberOfCores.size(); j++) {
			timings << "\"" << numberOfCores[j] << "\":" << cpuTimers[j][i].get();
			if (j != numberOfCores.size() - 1)
				timings << ",\n";
		}
		timings << "\n}";
		if (i != imageSizes.size() - 1)
			timings << ",\n";
	}
	timings << "\n}\n";
	timings << "\n}\n";

	return 0;
}