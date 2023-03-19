#include <iostream>
#include <stdio.h>
#include <time.h>

__global__ void renderVolume() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
    renderVolume<<<1,1>>>();
    return 0;
}