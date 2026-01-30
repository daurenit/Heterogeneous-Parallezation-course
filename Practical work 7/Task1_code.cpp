#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

// Макрос для проверки ошибок CUDA API
#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

/*
  CUDA-ядро для редукции (суммирования) элементов массива.

  Каждый блок обрабатывает 2 * blockDim.x элементов:
  - каждый поток загружает до двух элементов из глобальной памяти,
  - локальная сумма сохраняется в shared memory,
  - далее выполняется параллельная редукция внутри блока.
*/
__global__ void reduce_kernel(const float* input, float* blockSums, int n) {
    // Разделяемая память для частичных сумм внутри блока
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Загрузка данных из глобальной памяти
    float sum = 0.0f;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];

    // Запись локальной суммы в shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Параллельная редукция внутри блока
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Первый поток блока записывает сумму блока в глобальную память
    if (tid == 0)
        blockSums[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 1 << 20;        // Размер массива (1048576 элементов)
    const int blockSize = 256;   // Размер блока потоков

    // Инициализация входного массива на CPU
    std::vector<float> h_input(N, 1.0f);

    // ---------------- CPU редукция ----------------
    auto c1 = std::chrono::high_resolution_clock::now();
    float cpuSum = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
    auto c2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c2 - c1).count();

    // ---------------- GPU память ----------------
    float *d_input, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));

    // Каждый блок обрабатывает 2 * blockSize элементов
    int gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);
    CUDA_CHECK(cudaMalloc(&d_partial, gridSize * sizeof(float)));

    // Копирование данных на GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------- Замер времени GPU ----------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Запуск CUDA-ядра
    reduce_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_partial, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // ---------------- Сбор результатов ----------------
    std::vector<float> h_partial(gridSize);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial,
                          gridSize * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Финальная редукция частичных сумм на CPU
    float gpuSum = std::accumulate(h_partial.begin(), h_partial.end(), 0.0f);

    // ---------------- Вывод результатов ----------------
    std::cout << "N = " << N << "\n";
    std::cout << "CPU sum = " << cpuSum << "\n";
    std::cout << "GPU sum = " << gpuSum << "\n";
    std::cout << "Check = " << ((fabs(cpuSum - gpuSum) < 1e-3) ? "OK" : "ERROR") << "\n\n";
    std::cout << "CPU time = " << cpu_ms << " ms\n";
    std::cout << "GPU kernel time = " << gpu_ms << " ms\n";

    // Освобождение ресурсов
    cudaFree(d_input);
    cudaFree(d_partial);

    return 0;
}