#include <iostream>
#include <cmath>
#include <algorithm>
#include <stb_image_write.h>

#define IMAGE_SIZE 1080
#define CHANNELS 1
#define BLOCK_SIZE 16

__global__ void renderVolume(unsigned char *image) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

	image[IMAGE_SIZE * row + col] = 255;
}

int main() {
    dim3 gridDim(std::ceil(IMAGE_SIZE / BLOCK_SIZE), std::ceil(IMAGE_SIZE / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    
    unsigned char *image, *d_image;
    image = (unsigned char *)malloc(IMAGE_SIZE * IMAGE_SIZE * CHANNELS * sizeof(unsigned char));
    cudaMalloc(&d_image, IMAGE_SIZE * IMAGE_SIZE * CHANNELS * sizeof(unsigned char));
    renderVolume<<<gridDim, blockDim>>>(d_image);
    cudaMemcpy(image, d_image, IMAGE_SIZE * IMAGE_SIZE * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    
    stbi_write_png("test.png", IMAGE_SIZE, IMAGE_SIZE, CHANNELS, image, IMAGE_SIZE * CHANNELS);

    free(image);
    return 0;
}