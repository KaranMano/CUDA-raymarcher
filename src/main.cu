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
#include <cuda_runtime.h>
#include <future>

#include "core/camera.h"
#include "core/scene.h"
#include "core/vector.h"
#include "object/sphere.h"
#include "material/material.h"
#include "material/phong.h"
#include "material/volume.h"
#include "main.h"
#include <random>

#define IMAGE_SIZE 1080
#define CHANNELS 3
#define BLOCK_SIZE 16

// 1pt corresponds to 1m
unsigned char clamp(int value) {
	if (value > 255)
		return 255;
	else if (value < 0)
		return 0;
	else
		return value;
}

Vector renderPixel(int row, int col, const Scene& scene) {
	Ray ray = scene.camera().ray(row, col);
	scene.cast(ray);
	if (ray.intersected() == nullptr)
		return scene.background();
	else {
		return ray.intersected()->color(ray, scene);
	}
}

void renderScene(unsigned char *image, const Scene& scene, int offset, int band) {
	for (int row = 0; row < scene.camera().height(); row++)
		for (int col = offset; col < offset + band; col++)
		{
			int samples = 1;
			Vector color;
			for (int i = 0; i <= samples; i++) {
				color += renderPixel(row, col, scene);
			}
			color /= samples;
			image[(col + scene.camera().width() * row) * 3 + 0] = clamp(color.x * 255);
			image[(col + scene.camera().width() * row) * 3 + 1] = clamp(color.y * 255);
			image[(col + scene.camera().width() * row) * 3 + 2] = clamp(color.z * 255);
		}
}

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

	std::shared_ptr<Material> purple(new Phong({ 0.1f, 0.4f, 0.5f }));
	std::shared_ptr<Material> blue(new Phong({ 0.243f, 0.752f, 0.678f }));
	std::shared_ptr<Material> volume(new Volume({ 1.0f, 1.0f, 1.0f }));

	Scene scene;
	scene.add(std::shared_ptr<Object>(new Sphere({ 0.0f, 0.0f, -10.0f }, 3.0f, volume, true)));
	scene.add(std::shared_ptr<Object>(new Sphere({ -5.0f, 5.0f, -10.0f }, 2.0f, purple, false)));
	scene.add(std::shared_ptr<Object>(new Sphere({ 1.0f, 1.0f, -4.0f }, 0.5f, blue, false)));
	scene.add(std::shared_ptr<Light>(new Light({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 10.0)));

	unsigned char *image, *d_image;
	image = (unsigned char *)malloc(scene.camera().width() * scene.camera().height() * CHANNELS * sizeof(unsigned char));

	bool render = false;
	int spans = 10;
	int currSpan = 0;
	int steps = 8;
	int band = scene.camera().width() / (steps * spans);
	GLuint texture;
	bool isAvailable = false;
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::DockSpaceOverViewport();

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
			if (ImGui::Button("render")) {
				render = true;
			}
			ImGui::End();
		}
		if (render) {
			std::cout << "rendering volume\n";
			int offset = currSpan * ((float)scene.camera().width() / spans);
			std::vector<std::future<void>> jobs;
			for (int i = 0; i < steps; i++, offset += band) {
				jobs.emplace_back(std::async(std::launch::async, renderScene, image, scene, offset, band));
			}
			for (auto& job : jobs)
				job.get();

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
			currSpan++;
			stbi_write_png("render.png", scene.camera().width(), scene.camera().height(), CHANNELS, image, scene.camera().width() * CHANNELS);
			if (currSpan == spans) {
				currSpan = 0;
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
	return 0;
}