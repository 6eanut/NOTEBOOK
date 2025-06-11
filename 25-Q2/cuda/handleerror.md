## 错误处理

与在任何应用程序中一样，加速 CUDA 代码中的错误处理同样至关重要。即便不是大多数，也有许多 CUDA 函数（例如，[内存管理函数](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY)）会返回类型为 `cudaError_t` 的值，该值可用于检查调用函数时是否发生错误。以下是对调用 `cudaMallocManaged` 函数执行错误处理的示例：

```cpp
cudaError_t err;
err = cudaMallocManaged(&a, N)                    // Assume the existence of `a` and `N`.

if (err != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
{
  printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}
```

启动定义为返回 `void` 的核函数后，将不会返回类型为 `cudaError_t` 的值。为检查启动核函数时是否发生错误（例如，如果启动配置错误），CUDA 提供 `cudaGetLastError` 函数，该函数会返回类型为 `cudaError_t` 的值。

```cpp
/*
 * This launch should cause an error, but the kernel itself
 * cannot return it.
 */

someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.

cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
  printf("Error: %s\n", cudaGetErrorString(err));
}
```

最后，为捕捉异步错误（例如，在异步核函数执行期间），请务必检查后续同步 CUDA 运行时 API 调用所返回的状态（例如 `cudaDeviceSynchronize`）；如果之前启动的其中一个核函数失败，则将返回错误。

### 练习：添加错误处理

目前，[add-error-handling.cu](code\add-error-handling.cu) 会编译、运行并打印已加倍失败的数组元素。不过，该程序不会指明其中是否存在任何错误。重构应用程序以处理 CUDA 错误，以便您可以了解程序出现的问题并进行有效调试。您将需要调查在调用 CUDA 函数时可能出现的同步错误，以及在执行 CUDA 核函数时可能出现的异步错误。如您遇到问题，请参阅 [解决方案](code\add-error-handling-right.cu)。

### CUDA错误处理功能

创建一个包装 CUDA 函数调用的宏对于检查错误十分有用。以下是一个宏示例，您可以在余下练习中随时使用：

```cpp
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{

/*
 * The macro can be wrapped around any function returning
 * a value of type `cudaError_t`.
 */

  checkCuda( cudaDeviceSynchronize() )
}
```
