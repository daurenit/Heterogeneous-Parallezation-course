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


// Kernel 1: scan внутри блока + запись суммы блока 
__global__ void scan_block_inclusive(const int* __restrict__ d_in,
                                     int* __restrict__ d_out,
                                     int* __restrict__ d_blockSums,
                                     int n) {
  extern __shared__ int temp[]; // shared memory: 2*blockDim.x элементов

  int tid = threadIdx.x;
  int blockStart = 2 * blockDim.x * blockIdx.x;

  // Глобальные индексы двух элементов на поток
  int i1 = blockStart + tid;
  int i2 = blockStart + tid + blockDim.x;

  // Загружаем в shared memory (если выходим за границы — кладём 0)
  temp[tid] = (i1 < n) ? d_in[i1] : 0;
  temp[tid + blockDim.x] = (i2 < n) ? d_in[i2] : 0;

  __syncthreads();

  // Blelloch scan: up-sweep (reduce)
  // На этом шаге строим дерево сумм: (0+1), (2+3), потом (0..3) и так далее
  for (int offset = 1; offset < 2 * blockDim.x; offset <<= 1) {
    int idx = (tid + 1) * offset * 2 - 1;
    if (idx < 2 * blockDim.x) {
      temp[idx] += temp[idx - offset];
    }
    __syncthreads();
  }

  // В конце up-sweep temp[last] содержит сумму всего блока
  int last = 2 * blockDim.x - 1;
  int blockSum = temp[last];

  // Down-sweep (превращаем в exclusive scan)
  // Для down-sweep классически кладём 0 в последний элемент
  if (tid == 0) temp[last] = 0;
  __syncthreads();

  for (int offset = blockDim.x; offset > 0; offset >>= 1) {
    int idx = (tid + 1) * offset * 2 - 1;
    if (idx < 2 * blockDim.x) {
      int t = temp[idx - offset];
      temp[idx - offset] = temp[idx];
      temp[idx] += t;
    }
    __syncthreads();
  }

  // Сейчас temp[] - exclusive scan.
  // Чтобы получить inclusive scan: out[i] = exclusive[i] + in[i]
  if (i1 < n) d_out[i1] = temp[tid] + d_in[i1];
  if (i2 < n) d_out[i2] = temp[tid + blockDim.x] + d_in[i2];

  // Сумму блока сохраняем в blockSums (один поток на блок)
  if (d_blockSums && tid == 0) {
    d_blockSums[blockIdx.x] = blockSum;
  }
}

// Kernel 2: добавить оффсет блока ко всем элементам блока 
__global__ void add_block_offsets(int* __restrict__ d_out,
                                  const int* __restrict__ d_scannedBlockSums,
                                  int n) {
  int tid = threadIdx.x;
  int blockStart = 2 * blockDim.x * blockIdx.x;

  int i1 = blockStart + tid;
  int i2 = blockStart + tid + blockDim.x;

  // Для блока 0 оффсет = 0, для блока k оффсет = scannedBlockSums[k-1]
  int offset = (blockIdx.x == 0) ? 0 : d_scannedBlockSums[blockIdx.x - 1];

  if (i1 < n) d_out[i1] += offset;
  if (i2 < n) d_out[i2] += offset;
}

// Хост-функция: многоуровневый scan blockSums
void scan_block_sums_recursive(int* d_blockSums, int numBlocks) {
  // Если всего 1 блок — уже готово
  if (numBlocks <= 1) return;

  // Каждый блок сканирует 2*block элементов.
  // Для blockSums обычно numBlocks N/(2*block), это относительно мало
  const int block = 256; // 512 элементов на блокSums-скан (2*256)
  int elementsPerBlock = 2 * block;
  int grid = (numBlocks + elementsPerBlock - 1) / elementsPerBlock;

  int* d_nextLevelSums = nullptr;
  if (grid > 1) {
    CUDA_CHECK(cudaMalloc(&d_nextLevelSums, grid * sizeof(int)));
  }

  // Промежуточный массив для результата скана blockSums
  int* d_scanned = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scanned, numBlocks * sizeof(int)));

  // scan blockSums -> d_scanned, и собираем суммы блоков следующего уровня
  scan_block_inclusive<<<grid, block, elementsPerBlock * sizeof(int)>>>(
      d_blockSums, d_scanned, d_nextLevelSums, numBlocks);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Если есть следующий уровень - рекурсивно сканируем его
  if (grid > 1) {
    scan_block_sums_recursive(d_nextLevelSums, grid);

    // Добавляем оффсеты верхнего уровня к d_scanned (как к обычному output)
    add_block_offsets<<<grid, block>>>(d_scanned, d_nextLevelSums, numBlocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_nextLevelSums));
  }

  // Копируем обратно в d_blockSums (теперь там префиксные суммы блоков)
  CUDA_CHECK(cudaMemcpy(d_blockSums, d_scanned, numBlocks * sizeof(int), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaFree(d_scanned));
}

// CPU scan (inclusive) 
void cpu_scan_inclusive(const std::vector<int>& in, std::vector<int>& out) {
  long long running = 0;
  for (size_t i = 0; i < in.size(); i++) {
    running += in[i];
    out[i] = (int)running;
  }
}

// Проверка нескольких точек 
bool check_some(const std::vector<int>& cpu, const std::vector<int>& gpu) {
  for (int idx : {0, 1, 2, 123, 999999}) {
    if (cpu[idx] != gpu[idx]) return false;
  }
  return true;
}

int main() {
  const int N = 1'000'000;

  // Данные 
  std::vector<int> h_in(N), h_cpu(N), h_gpu(N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(0, 100);
  for (int i = 0; i < N; i++) h_in[i] = dist(rng);

  //CPU scan + время
  auto t1 = std::chrono::high_resolution_clock::now();
  cpu_scan_inclusive(h_in, h_cpu);
  auto t2 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

  // GPU память
  int *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  // Параметры запуска основного scan
  const int block = 512;               // 512 потоков
  const int elementsPerBlock = 2 * block; // 1024 элемента на блок
  int grid = (N + elementsPerBlock - 1) / elementsPerBlock;

  int* d_blockSums = nullptr;
  if (grid > 1) {
    CUDA_CHECK(cudaMalloc(&d_blockSums, grid * sizeof(int)));
  }

  // Тайминг GPU (только вычисления на GPU)
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  // 1) scan по блокам + суммы блоков
  scan_block_inclusive<<<grid, block, elementsPerBlock * sizeof(int)>>>(
      d_in, d_out, d_blockSums, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 2) scan массива сумм блоков + добавление оффсетов
  if (grid > 1) {
    scan_block_sums_recursive(d_blockSums, grid);

    add_block_offsets<<<grid, block>>>(d_out, d_blockSums, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float gpu_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Забираем результат
  CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = check_some(h_cpu, h_gpu);

  std::cout << "N = " << N << "\n";
  std::cout << "Check = " << (ok ? "OK" : "ERROR") << "\n\n";
  std::cout << "CPU time (sequential scan) = " << cpu_ms << " ms\n";
  std::cout << "GPU time (shared memory scan) = " << gpu_ms << " ms\n";

  //Освобождение памяти
  if (d_blockSums) CUDA_CHECK(cudaFree(d_blockSums));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  return 0;
}