#include <iostream>
#include <cmath>
#include <algorithm>
#include <stb_image_write.h>
#include <cuda_runtime.h>

#include "core/camera.h"
#include "core/vector.h"

#define IMAGE_SIZE 1080
#define CHANNELS 3
#define BLOCK_SIZE 16

__global__ void renderVolume(unsigned char *image, Camera cam) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    Ray ray = cam.getRay(row, col);
    image[(col + cam.imageCols * row) * 3 + 0] = 255;//ray.origin.x;
    image[(col + cam.imageCols * row) * 3 + 1] = ray.origin.y;
    image[(col + cam.imageCols * row) * 3 + 2] = ray.origin.z;
}

int main() {
    std::cout << "rendering volume\n";
    Vector position(0.0f, 0.0f, 1.0f); 
    Vector view(0.0f, 0.0f, -1.0f);
    Vector up(0.0f, 1.0f, 0.0f);
    Camera cam(
        position, 
        view, 
        up, 
        100,
        IMAGE_SIZE,
        IMAGE_SIZE
    );

    dim3 gridDim(std::ceil(IMAGE_SIZE / BLOCK_SIZE), std::ceil(IMAGE_SIZE / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    
    unsigned char *image, *d_image;
    image = (unsigned char *)malloc(IMAGE_SIZE * IMAGE_SIZE * CHANNELS * sizeof(unsigned char));
    cudaMalloc(&d_image, IMAGE_SIZE * IMAGE_SIZE * CHANNELS * sizeof(unsigned char));
    renderVolume<<<gridDim, blockDim>>>(d_image, cam);
    cudaMemcpy(image, d_image, IMAGE_SIZE * IMAGE_SIZE * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    
    stbi_write_png("test.png", IMAGE_SIZE, IMAGE_SIZE, CHANNELS, image, IMAGE_SIZE * CHANNELS);

    free(image);
    return 0;
}