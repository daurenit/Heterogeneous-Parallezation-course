#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while(0)

/*
  
  Общая идея:
  1) Каждый поток берёт один элемент input[i] и добавляет его в общую сумму
  2) Так как несколько потоков пишут в одну переменную, нужен atomicAdd
  3) atomicAdd по int работает напрямую
*/

// GPU kernel: суммирование через global memory + atomicAdd
__global__ void sum_global_atomic(const int* __restrict__ input, int* __restrict__ out_sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Проверяем границы массива
  if (i < n) {
    // Атомарно прибавляем input[i] к out_sum[0] в global memory
    atomicAdd(out_sum, input[i]);
  }
}

// CPU: последовательная сумма
long long cpu_sum(const std::vector<int>& a) {
  long long s = 0;
  for (int x : a) s += x;
  return s;
}

int main() {
  const int N = 100000;

  // 1) Генерируем данные на CPU
  std::vector<int> h_in(N);
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(0, 100); // маленькие числа, чтобы не переполнить int
  for (int i = 0; i < N; i++) h_in[i] = dist(rng);

  // 2) CPU версия: считаем сумму и меряем время
  auto t1 = std::chrono::high_resolution_clock::now();
  long long cpu = cpu_sum(h_in);
  auto t2 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

  // 3) Выделяем память на GPU
  int *d_in = nullptr;
  int *d_sum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(int)));

  // 4) Копируем входные данные на GPU
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  // 5) Обнуляем сумму на GPU
  CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(int)));

  // 6) Настраиваем сетку, блок
  int block = 256;
  int grid  = (N + block - 1) / block;

  // 7) Замеряем время GPU kernel через cudaEvent
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  sum_global_atomic<<<grid, block>>>(d_in, d_sum, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float gpu_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // 8) Забираем результат с GPU на CPU
  int gpu_int = 0;
  CUDA_CHECK(cudaMemcpy(&gpu_int, d_sum, sizeof(int), cudaMemcpyDeviceToHost));

  long long gpu = static_cast<long long>(gpu_int);

  // 9) Сравнение результатов
  bool ok = (cpu == gpu);

  std::cout << "N = " << N << "\n";
  std::cout << "CPU sum = " << cpu << "\n";
  std::cout << "GPU sum = " << gpu << "\n";
  std::cout << "Check = " << (ok ? "OK" : "ERROR") << "\n\n";

  std::cout << "CPU time (sequential) = " << cpu_ms << " ms\n";
  std::cout << "GPU kernel time (global + atomic) = " << gpu_ms << " ms\n";

  // 10) Освобождаем память
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_sum));

  return 0;
}