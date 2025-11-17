# GPGPU-SIM

```shell
# cuda toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
export CUDA_INSTALL_PATH=/usr/local/cuda-11.8
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# gpgpu-sim
sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev doxygen graphviz python-pmw python-ply python-numpy libpng12-dev python-matplotlib libxi-dev libxmu-dev libglut3-dev
git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution.git
cd gpgpu-sim_distribution
source setup_environment
make

# demo
/* file: vector_add.cu */
#include <stdio.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main() {
    int n = 16;
    size_t bytes = n * sizeof(float);

    float h_a[16], h_b[16], h_c[16];

    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    float *d_a, *d_b, *d_c;

    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    vector_add<<<1, 16>>>(d_a, d_b, d_c, n);

    // 关键：检查 kernel 运行错误
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    printf("Result: ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


# run
nvcc --cudart shared -o vector_add vector_add.cu
ldd vector_add
cp pathto/gpgpu-sim_distribution/configs/tested-cfgs/SM86_RTX3070/* . -r
./vector_add
```
