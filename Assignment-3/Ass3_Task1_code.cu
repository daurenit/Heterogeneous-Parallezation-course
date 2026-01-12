#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while(0)

// Версия 1: работаем только через глобальную память (каждый поток читает/пишет напрямую).
__global__ void mul_global(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * k;
}

// Версия 2: используем разделяемую (shared) память.
// Тут смысл в демонстрации: блок сначала копирует свой кусок в shared,
// потом умножает и пишет результат обратно в global.
// На такой операции ускорение может быть небольшим (или вообще не быть),
// но требование "использовать shared memory" выполняется честно.
__global__ void mul_shared(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k,
                           int n) {
  extern __shared__ float s[];      // shared-буфер на блок (размер задаём при запуске kernel)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int t = threadIdx.x;

  if (i < n) s[t] = in[i];          // global -> shared
  __syncthreads();                  // ждём, пока все потоки блока загрузят данные

  if (i < n) out[i] = s[t] * k;     // shared -> вычисление -> global
}

// Для GPU корректнее мерить время через cudaEvent (chrono будет считать ещё и синхронизации/запуски неявно).
float time_kernel_global(const float* d_in, float* d_out, float k, int n, int block) {
  int grid = (n + block - 1) / block;

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  CUDA_CHECK(cudaEventRecord(s));
  mul_global<<<grid, block>>>(d_in, d_out, k, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms;
}

float time_kernel_shared(const float* d_in, float* d_out, float k, int n, int block) {
  int grid = (n + block - 1) / block;
  size_t shmem_bytes = block * sizeof(float);  // по одному float на поток

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  CUDA_CHECK(cudaEventRecord(s));
  mul_shared<<<grid, block, shmem_bytes>>>(d_in, d_out, k, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms;
}

int main() {
  const int N = 1'000'000;
  const float k = 2.5f;

  // Заполняем входной массив на CPU (чтобы данные были не одинаковые).
  std::vector<float> h_in(N), h_out(N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 100.0f);
  for (int i = 0; i < N; i++) h_in[i] = dist(rng);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Чуть “прогреваем” GPU (первый запуск часто медленнее из-за инициализации).
  {
    int block = 256;
    int grid = (N + block - 1) / block;
    mul_global<<<grid, block>>>(d_in, d_out, k, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  int block = 256;  // базовый размер блока (можно менять, но в этой задаче не требуется)

  float t_global = time_kernel_global(d_in, d_out, k, N, block);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Быстрая проверка нескольких элементов, чтобы убедиться, что вычисления правильные.
  bool ok1 = true;
  for (int idx : {0, 1, 123, 999999}) {
    float ref = h_in[idx] * k;
    if (std::fabs(h_out[idx] - ref) > 1e-4f) ok1 = false;
  }

  float t_shared = time_kernel_shared(d_in, d_out, k, N, block);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok2 = true;
  for (int idx : {0, 1, 123, 999999}) {
    float ref = h_in[idx] * k;
    if (std::fabs(h_out[idx] - ref) > 1e-4f) ok2 = false;
  }

  std::cout << "N = " << N << "\n";
  std::cout << "Block size = " << block << "\n";
  std::cout << "Global memory kernel time: " << t_global << " ms\n";
  std::cout << "Shared memory kernel time: " << t_shared << " ms\n";
  std::cout << "Check global: " << (ok1 ? "OK" : "ERROR") << "\n";
  std::cout << "Check shared: " << (ok2 ? "OK" : "ERROR") << "\n";

  if (t_shared < t_global) {
    std::cout << "Conclusion: shared version was faster on this run.\n";
  } else {
    std::cout << "Conclusion: global version was faster or similar on this run (overheads may dominate).\n";
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}