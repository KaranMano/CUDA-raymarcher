#pragma once
#include <vector>
#include <future>
#include <cuda_runtime.h>

#include "core/camera.h"
#include "core/scene.h"
#include "core/vector.h"
#include "object/sphere.h"
#include "material/material.h"
#include "material/phong.h"
#include "material/volume.h"

#define CHANNELS 3

__host__ __device__
unsigned char clamp(int value);

__host__ __device__
Vector renderPixel(int row, int col, Scene& scene);

__global__
void renderKernel(unsigned char* image, Scene* scene);

__global__
void launchKernel(unsigned char* image, Scene* scene, int blockSize);

__global__
void setupKernel(Scene* scene);

__global__
void cleanupKernel(Scene* scene);

void renderScene(unsigned char *image, Scene& scene, int offset, int band);

void cpuKernel(unsigned char* image, Scene &scene, int steps);

void checkCudaErrors(cudaError_t err);