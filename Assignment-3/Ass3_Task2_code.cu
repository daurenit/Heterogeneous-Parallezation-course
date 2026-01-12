#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Макрос для проверки CUDA-вызовов
// Если где-то ошибка (malloc, memcpy, kernel launch и т.д.) - сразу печатаем причину и выходим
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while(0)

// CUDA kernel: поэлементное сложение векторов a и b -> c
// Каждый поток отвечает за один индекс i
// __restrict__ подсказка компилятору: указатели не пересекаются, можно агрессивнее оптимизировать
__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c,
                        int n) {
  // Глобальный индекс потока:
  // blockIdx.x - номер блока
  // blockDim.x - число потоков в блоке (block size)
  // threadIdx.x - номер потока внутри блока
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Обязательно проверяем границу, т.к. grid обычно округляем вверх
  // и "лишние" потоки могут выходить за n
  if (i < n) c[i] = a[i] + b[i];
}

// Функция измерения времени только KERNEL части через cudaEvent
// Важно: chrono на CPU может включить лишние задержки/синхронизации и будет менее корректно
float time_add_kernel(const float* d_a, const float* d_b, float* d_c, int n, int block) {
  // Grid size выбираем так, чтобы покрыть все n элементов
  int grid = (n + block - 1) / block;

  // CUDA events - стандартный способ точного замера времени на GPU
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  // Ставим "старт", запускаем kernel, ставим "стоп"
  CUDA_CHECK(cudaEventRecord(s));
  vec_add<<<grid, block>>>(d_a, d_b, d_c, n);

  // Проверяем, что запуск kernel прошёл без ошибок (например, неправильная конфигурация)
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(e));

  // Ждём, пока kernel реально закончится, иначе время будет "неполным"
  CUDA_CHECK(cudaEventSynchronize(e));

  // Получаем время между событиями в миллисекундах
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

  // Чистим ресурсы
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms;
}

// Лёгкая проверка правильности: не проверяем весь массив (это долго), а проверяем несколько фиксированных индексов
bool check_some(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c) {
  for (int idx : {0, 1, 123, 999999}) {
    float ref = a[idx] + b[idx];
    if (std::fabs(c[idx] - ref) > 1e-4f) return false;
  }
  return true;
}

int main() {
  // По заданию работаем с большим массивом. Здесь N = 1 000 000
  const int N = 1'000'000;

  // Host (CPU) массивы
  // h_a, h_b - входные
  // h_c - выходной, сюда скопируем результат с GPU
  std::vector<float> h_a(N), h_b(N), h_c(N);

  // Заполняем входные данные псевдослучайными значениями, чтобы тест был "реальный"
  // Фиксируем seed, чтобы результаты были воспроизводимы
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 100.0f);
  for (int i = 0; i < N; i++) {
    h_a[i] = dist(rng);
    h_b[i] = dist(rng);
  }

  // Device (GPU) указатели
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  // Выделяем память на GPU под три массива
  CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

  // Копируем входные массивы с CPU -> GPU
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Минимум три размера блока - как требует задание
  // Идея: посмотреть, как block size влияет на скорость kernel
  std::vector<int> blocks = {128, 256, 512};

  // "Прогрев" GPU.
  // Первый запуск kernel часто медленнее из-за lazy init (драйвер, контекст, кеши)
  // Поэтому делаем один прогон без замера, чтобы результаты замеров были честнее
  {
    int block = 256;
    int grid = (N + block - 1) / block;
    vec_add<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::cout << "N = " << N << "\n";

  // Тут будем хранить лучший результат (минимальное время) и какой block size его дал
  float best = 1e9f;
  int bestBlock = -1;

  // Основной цикл эксперимента по разным размерам блока
  for (int block : blocks) {
    // 1) Замеряем время выполнения kernel
    float ms = time_add_kernel(d_a, d_b, d_c, N, block);

    // 2) Копируем результат обратно на CPU для проверки
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 3) Проверяем корректность на нескольких элементах
    bool ok = check_some(h_a, h_b, h_c);

    // 4) Печатаем результат по этому block size
    std::cout << "Block size = " << block << "\n";
    std::cout << "Kernel time = " << ms << " ms\n";
    std::cout << "Check = " << (ok ? "OK" : "ERROR") << "\n\n";

    // 5) Обновляем "лучший" вариант, если этот оказался быстрее
    if (ms < best) { best = ms; bestBlock = block; }
  }

  // Итог по эксперименту
  std::cout << "Best block size on this run = " << bestBlock << "\n";
  std::cout << "Best time = " << best << " ms\n";

  // Освобождаем память на GPU
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}