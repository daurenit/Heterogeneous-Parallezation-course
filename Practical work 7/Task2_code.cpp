#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

/*
  Inclusive scan (prefix sum) внутри одного блока.
  Каждый блок обрабатывает blockDim.x элементов.
  Для учебной задачи предполагается, что N кратно blockDim.x.
*/

__global__ void scan_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    // Загрузка в shared memory
    if (i < n)
        sdata[tid] = input[i];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    // Inclusive scan (Hillis–Steele)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = 0.0f;
        if (tid >= offset)
            val = sdata[tid - offset];
        __syncthreads();
        sdata[tid] += val;
        __syncthreads();
    }

    if (i < n)
        output[i] = sdata[tid];
}

float cpu_scan(const std::vector<float>& in, std::vector<float>& out) {
    auto t1 = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;
    for (size_t i = 0; i < in.size(); i++) {
        sum += in[i];
        out[i] = sum;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

int main() {
    const int N = 1 << 20;        // 1 048 576
    const int blockSize = 256;

    std::vector<float> h_in(N, 1.0f);
    std::vector<float> h_cpu(N), h_gpu(N);

    // CPU
    double cpu_ms = cpu_scan(h_in, h_cpu);

    // GPU memory
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    scan_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i : {0, 1, 2, 123, N - 1}) {
        if (fabs(h_cpu[i] - h_gpu[i]) > 1e-4) {
            ok = false;
            break;
        }
    }

    std::cout << "N = " << N << "\n";
    std::cout << "Check = " << (ok ? "OK" : "ERROR") << "\n";
    std::cout << "CPU time = " << cpu_ms << " ms\n";
    std::cout << "GPU time = " << gpu_ms << " ms\n";

    cudaFree(d_in);
    cudaFree(d_out);
}