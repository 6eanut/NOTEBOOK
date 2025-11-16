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
/* file: hello.cu */
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    int c;
    int *dev_c;
    cudaMalloc((void **)&dev_c, sizeof(int));
    add<<<1, 1>>>(2, 7, dev_c);
    cudaMemcpy(&c, &dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("2 + 7 = %d\n", c);

    return 0;
}

# run
nvcc --cudart shared -o hello hello.cu
ldd hello
cp pathto/gpgpu-sim_distribution/configs/tested-cfgs/SM86_RTX3070/ . -r
./hello
```
