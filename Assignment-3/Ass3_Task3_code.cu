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

// Коалесцированный доступ: поток i читает/пишет элемент i
// Для warp это обычно подряд идущие адреса -> GPU объединяет транзакции памяти эффективнее
__global__ void coalesced_op(const float* __restrict__ in,
                             float* __restrict__ out,
                             float k,
                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * k;
}

// Некоалесцированный доступ: делаем "перемешанный" индекс
// Потоки warp читают не подряд, а через шаг (stride) -> больше транзакций, хуже пропускная способность
// Тут мы берём stride = 32 (примерно размер warp), чтобы эффект был заметнее
__global__ void noncoalesced_op(const float* __restrict__ in,
                                float* __restrict__ out,
                                float k,
                                int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = 32;

  // Перестановка индексов по "полосам":
  // tid 0..31 -> idx 0,32,64,... (не подряд)
  // tid 32..63 -> idx 1,33,65,... и т.д.
  int lane = tid % stride;
  int group = tid / stride;
  int idx = group + lane * ((n + stride - 1) / stride);

  if (idx < n) out[idx] = in[idx] * k;
}

// Замер времени kernel через cudaEvent.
template <typename Kernel, typename... Args>
float time_kernel(dim3 grid, dim3 block, size_t shmem, Kernel kfun, Args... args) {
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  CUDA_CHECK(cudaEventRecord(s));
  kfun<<<grid, block, shmem>>>(args...);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms;
}

// Быстрая проверка: несколько точек
bool check_some_mul(const std::vector<float>& in, const std::vector<float>& out, float k) {
  for (int idx : {0, 1, 123, 999999}) {
    float ref = in[idx] * k;
    if (std::fabs(out[idx] - ref) > 1e-3f) return false;
  }
  return true;
}

int main() {
  const int N = 1'000'000;
  const float k = 2.5f;
  const int blockSize = 256;

  std::vector<float> h_in(N), h_out(N);

  // Заполняем вход
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 100.0f);
  for (int i = 0; i < N; i++) h_in[i] = dist(rng);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(blockSize);
  dim3 grid((N + blockSize - 1) / blockSize);

  // Прогрев (чтобы первый запуск не портил картину)
  coalesced_op<<<grid, block>>>(d_in, d_out, k, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 1) Коалесцированный вариант
  float t_coal = time_kernel(grid, block, 0, coalesced_op, d_in, d_out, k, N);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok1 = check_some_mul(h_in, h_out, k);

  // 2) Некоалесцированный вариант
  float t_non = time_kernel(grid, block, 0, noncoalesced_op, d_in, d_out, k, N);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok2 = check_some_mul(h_in, h_out, k);

  std::cout << "N = " << N << "\n";
  std::cout << "Block size = " << blockSize << "\n";
  std::cout << "Coalesced kernel time: " << t_coal << " ms\n";
  std::cout << "Non-coalesced kernel time: " << t_non << " ms\n";
  std::cout << "Check coalesced: " << (ok1 ? "OK" : "ERROR") << "\n";
  std::cout << "Check non-coalesced: " << (ok2 ? "OK" : "ERROR") << "\n";

  if (t_coal < t_non) {
    std::cout << "Conclusion: coalesced access is faster on this run.\n";
  } else {
    std::cout << "Conclusion: times are similar or non-coalesced was not slower (depends on overheads/cache).\n";
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}