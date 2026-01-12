#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while(0)

// Берём одну из прошлых задач: поэлементное сложение двух массивов
// На ней удобно подбирать параметры block и grid
__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c,
                        int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// Чисто замер времени kernel через cudaEvent
// Важно: мерим только kernel (без memcpy) иначе сравнение размеров блока будет "шумным"
float time_add_kernel(const float* d_a, const float* d_b, float* d_c, int n, int block) {
  int grid = (n + block - 1) / block;

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  CUDA_CHECK(cudaEventRecord(s));
  vec_add<<<grid, block>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms;
}

// Быстрая проверка корректности по нескольким индексам
bool check_some(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c) {
  for (int idx : {0, 1, 123, 999999}) {
    float ref = a[idx] + b[idx];
    if (std::fabs(c[idx] - ref) > 1e-4f) return false;
  }
  return true;
}

int main() {
  const int N = 1'000'000;

  std::vector<float> h_a(N), h_b(N), h_c(N);

  // Делаем данные "не одинаковыми", чтобы было похоже на реальную нагрузку
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 100.0f);
  for (int i = 0; i < N; i++) {
    h_a[i] = dist(rng);
    h_b[i] = dist(rng);
  }

  // GPU память
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Прогрев: первый запуск часто медленнее из за инициализации контекста GPU
  {
    int block = 256;
    int grid = (N + block - 1) / block;
    vec_add<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // 1) Неоптимальная конфигурация (берём заведомо плохую: очень маленький блок)
  // На практике это даёт меньше параллелизма и хуже загрузку GPU
  int block_bad = 32;
  float t_bad = time_add_kernel(d_a, d_b, d_c, N, block_bad);

  // 2) Подбор "оптимальной" конфигурации:
  // Обычно проверяют несколько типичных размеров: 64, 128, 256, 512, 1024
  // Максимум потоков на блок зависит от GPU, но на T4 обычно до 1024
  std::vector<int> candidates = {64, 128, 256, 512, 1024};

  float best_t = std::numeric_limits<float>::max();
  int best_block = -1;

  for (int block : candidates) {
    // Если block слишком большой и не поддерживается, kernel может не запуститься
    // Поэтому сразу ловим ошибки через CUDA_CHECK внутри time_add_kernel
    float ms = time_add_kernel(d_a, d_b, d_c, N, block);

    if (ms < best_t) {
      best_t = ms;
      best_block = block;
    }
  }

  // Проверяем корректность именно на "лучшей" конфигурации
  {
    int grid = (N + best_block - 1) / best_block;
    vec_add<<<grid, best_block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = check_some(h_a, h_b, h_c);

    std::cout << "N = " << N << "\n\n";

    std::cout << "Non-optimal config:\n";
    std::cout << "  block = " << block_bad << "\n";
    std::cout << "  kernel time = " << t_bad << " ms\n\n";

    std::cout << "Optimized config (best among candidates):\n";
    std::cout << "  block = " << best_block << "\n";
    std::cout << "  kernel time = " << best_t << " ms\n\n";

    std::cout << "Check = " << (ok ? "OK" : "ERROR") << "\n";

    if (best_t < t_bad) {
      std::cout << "Conclusion: optimized configuration is faster on this run.\n";
    } else {
      std::cout << "Conclusion: difference is small or non-optimal was not slower (depends on noise/overheads).\n";
    }
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}