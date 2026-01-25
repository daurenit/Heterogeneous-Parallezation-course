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

/*Сравниваем 3 режима:
  1) CPU only: весь массив обрабатывается последовательно на CPU
  2) GPU only: весь массив обрабатывается одним kernel на GPU
  3) Hybrid: первая часть массива (0..split-1) на CPU,
             вторая часть (split..N-1) на GPU одновременно
*/

// GPU kernel: простая поэлементная обработка
__global__ void process_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               float k,
                               int n,
                               int offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = offset + tid;
  if (i < n) out[i] = in[i] * k + 1.0f;
}

// CPU обработка участка [0, end)
void cpu_process_range(const std::vector<float>& in,
                       std::vector<float>& out,
                       float k,
                       int end) {
  for (int i = 0; i < end; i++) out[i] = in[i] * k + 1.0f;
}

// CPU проверка нескольких элементов
bool check_some(const std::vector<float>& ref, const std::vector<float>& got) {
  for (int idx : {0, 1, 123, 999999}) {
    if (std::fabs(ref[idx] - got[idx]) > 1e-4f) return false;
  }
  return true;
}

int main() {
  const int N = 1'000'000;
  const float k = 2.5f;

  //Разделение для гибрида: 50% CPU и 50% GPU
  const int split = N / 2;

  // Данные на CPU
  std::vector<float> h_in(N), out_cpu(N), out_gpu(N), out_hybrid(N), out_ref(N);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 100.0f);
  for (int i = 0; i < N; i++) h_in[i] = dist(rng);

  // Эталон (CPU) для проверки корректности
  for (int i = 0; i < N; i++) out_ref[i] = h_in[i] * k + 1.0f;

  // CPU only 
  auto c1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) out_cpu[i] = h_in[i] * k + 1.0f;
  auto c2 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(c2 - c1).count();

  //  GPU memory 
  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  //  GPU only 
  {
    int block = 256;
    int grid  = (N + block - 1) / block;

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    process_kernel<<<grid, block>>>(d_in, d_out, k, N, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, s, e));

    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));

    CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = check_some(out_ref, out_gpu);

    std::cout << "N = " << N << "\n\n";
    std::cout << "[CPU only]\n";
    std::cout << "Time = " << cpu_ms << " ms\n\n";
    std::cout << "[GPU only]\n";
    std::cout << "Kernel time = " << gpu_ms << " ms\n";
    std::cout << "Check = " << (ok ? "OK" : "ERROR") << "\n\n";
  }

  // Hybrid (CPU + GPU parallel) 
  // Для параллельности используем:
  //-CPU делает первую половину
  //-GPU запускает kernel на вторую половину в отдельном stream
  //-копирование результата GPU-части делаем async и синхронизируем в конце
  float hybrid_gpu_ms = 0.0f;
  double hybrid_total_ms = 0.0;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Буфер для второй половины на хосте (pinned), чтобы cudaMemcpyAsync было корректным
  float* h_pinned_second = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_pinned_second, (N - split) * sizeof(float)));

  auto h1 = std::chrono::high_resolution_clock::now();

  //1) CPU часть: [0..split)
  cpu_process_range(h_in, out_hybrid, k, split);

  // 2) GPU часть: [split..N)
  int block = 256;
  int n2 = N - split;
  int grid2 = (n2 + block - 1) / block;

  cudaEvent_t s2, e2;
  CUDA_CHECK(cudaEventCreate(&s2));
  CUDA_CHECK(cudaEventCreate(&e2));

  CUDA_CHECK(cudaEventRecord(s2, stream));
  process_kernel<<<grid2, block, 0, stream>>>(d_in, d_out, k, N, split);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e2, stream));

  // Забираем вторую половину асинхронно в pinned buffer
  CUDA_CHECK(cudaMemcpyAsync(h_pinned_second,
                            d_out + split,
                            n2 * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            stream));

  // Ждём GPU stream
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaEventSynchronize(e2));
  CUDA_CHECK(cudaEventElapsedTime(&hybrid_gpu_ms, s2, e2));

  CUDA_CHECK(cudaEventDestroy(s2));
  CUDA_CHECK(cudaEventDestroy(e2));

  // Перекладываем pinned часть в общий out_hybrid
  for (int i = 0; i < n2; i++) out_hybrid[split + i] = h_pinned_second[i];

  auto h2 = std::chrono::high_resolution_clock::now();
  hybrid_total_ms = std::chrono::duration<double, std::milli>(h2 - h1).count();

  bool ok_h = check_some(out_ref, out_hybrid);

  std::cout << "[Hybrid CPU+GPU]\n";
  std::cout << "Total time = " << hybrid_total_ms << " ms\n";
  std::cout << "GPU kernel time (second half) = " << hybrid_gpu_ms << " ms\n";
  std::cout << "Check = " << (ok_h ? "OK" : "ERROR") << "\n\n";

  std::cout << "Conclusion: compare CPU-only, GPU-only kernel time, and Hybrid total time.\n";

  // Освобождение ресурсов
  CUDA_CHECK(cudaFreeHost(h_pinned_second));
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  return 0;
}