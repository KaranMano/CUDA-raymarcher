#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_stdlib.h>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stb_image_write.h>

#include "utils.h"
#include <random>

int main() {
	if (!glfwInit())
		return -1;

	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	GLFWwindow* window = glfwCreateWindow(500, 500, "image-morphing", NULL, NULL);

	if (!window) {
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	if (glewInit() != GLEW_OK)
		return -1;

	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	//Volume volume({ 1.0f, 1.0f, 1.0f });
	
	unsigned char *dump, *d_dump;
	size_t volumeSize = 256 * 256 * 128 * sizeof(unsigned char);
	dump = (unsigned char*)malloc(volumeSize);
	cudaMalloc((void **)&d_dump, volumeSize);
	std::ifstream volume("../data/engine_256x256x128_uint8.raw");
	volume.read((char*)dump, volumeSize);
	cudaMemcpy(d_dump, dump, volumeSize, cudaMemcpyHostToDevice);

	Scene scene(1080, 1080);
	scene.add((Object*)new Sphere({ 0.0f, 0.0f, -10.0f }, 6.0f, true), (Material*)new Volume(dump));
	scene.add(new Light({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 10.0));

	unsigned char *image, *d_image;
	image = (unsigned char *)malloc(scene.camera().width() * scene.camera().height() * CHANNELS * sizeof(unsigned char));

	bool render = false;
	int steps = 8;
	int band = scene.camera().width() / steps;
	std::vector<std::future<void>> jobs;
	GLuint texture;
	bool isAvailable = false, isGPU = false;
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::DockSpaceOverViewport();

		{
			ImGui::Begin("Controls");
			if (ImGui::Button("Save"))
				stbi_write_png("render.png", scene.camera().width(), scene.camera().height(), CHANNELS, image, scene.camera().width() * CHANNELS);
			ImGui::Checkbox("GPU", &isGPU);
			if (ImGui::Button("Render")) {
				memset(image, 0, scene.camera().width() * scene.camera().height()* CHANNELS);
				render = true;
				if (!isGPU) {
					for (int step = 0; step < steps; step++) {
						jobs.emplace_back(std::async(std::launch::async, renderScene, image, scene, step * band, band));
					}
				}
				else {
					Scene dummy(1080, 1080), *d_scene;
					cudaMalloc(&d_scene, sizeof(dummy));
					cudaMemcpy(d_scene, &dummy, sizeof(dummy), cudaMemcpyHostToDevice);
					cudaMalloc(&d_image, scene.camera().height()* scene.camera().width() * CHANNELS);
					
					setupKernel << <1, 1 >> > (d_scene, d_dump);
					checkCudaErrors(cudaDeviceSynchronize());
					launchKernel << <1, 1 >> > (d_image, d_scene, 16);
					checkCudaErrors(cudaDeviceSynchronize());
					cleanupKernel << <1, 1 >> > (d_scene);
					checkCudaErrors(cudaDeviceSynchronize());

					cudaMemcpy(image, d_image, scene.camera().height()* scene.camera().width() * CHANNELS, cudaMemcpyDeviceToHost);
					cudaFree(d_image);
					cudaFree(d_scene);

					if (isAvailable) {
						glDeleteTextures(1, &texture);
						isAvailable = false;
					}

					glGenTextures(1, &texture);
					glBindTexture(GL_TEXTURE_2D, texture);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
					glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scene.camera().width(), scene.camera().height(), 0, GL_RGB, GL_UNSIGNED_BYTE, image);
					glBindTexture(GL_TEXTURE_2D, 0);
					isAvailable = true;
					render = false;
				}
			}

			ImGui::End();
		}

		{
			ImGui::Begin("Result");
			if (isAvailable) {
				ImVec2 winSize = ImGui::GetWindowSize();
				float scaleFactor = std::min(winSize.y / scene.camera().height(), winSize.x / scene.camera().width());
				ImGui::Image(
					reinterpret_cast<void*>(static_cast<intptr_t>(texture)),
					ImVec2(scene.camera().width() * scaleFactor, scene.camera().height() * scaleFactor)
				);
			}
			ImGui::End();
		}
		if (render) {
			if (isAvailable) {
				glDeleteTextures(1, &texture);
				isAvailable = false;
			}

			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scene.camera().width(), scene.camera().height(), 0, GL_RGB, GL_UNSIGNED_BYTE, image);
			glBindTexture(GL_TEXTURE_2D, 0);

			isAvailable = true;
			bool isRenderComplete = true;
			for (auto &job : jobs) {
				using namespace std::chrono_literals;
				isRenderComplete &= (job.wait_for(0s) == std::future_status::ready);
			}
			if (isRenderComplete) {
				render = false;
			}
		}

		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
	}

	glfwTerminate();
	free(image);
	free(dump);
	cudaFree(d_dump);
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
	return 0;
}