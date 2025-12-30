#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace std;

// Настройки
static constexpr int CHUNK = 1024;   // размер подмассива, который сортирует один блок (должен быть степенью 2)
static constexpr int THREADS = 256;  // потоков в блоке для сортировки/слияния

// Проверка CUDA ошибок (чтобы не ловить "тихие" баги)
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) \
             << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

// Bitonic sort в shared memory для одного чанка
// Каждый блок берет один чанк длиной CHUNK
__global__ void sortChunksBitonic(const int* __restrict__ d_in, int* __restrict__ d_out, int n) {
    __shared__ int s[CHUNK];

    int chunkId = blockIdx.x;                 // какой чанк обрабатываем
    int base = chunkId * CHUNK;               // стартовый индекс чанка в глобальной памяти
    int tid = threadIdx.x;

    // Загружаем чанк в shared memory
    // Если вышли за границы n, кладем INT_MAX (как "пустышку", чтобы она ушла в конец при сортировке)
    for (int i = tid; i < CHUNK; i += blockDim.x) {
        int idx = base + i;
        s[i] = (idx < n) ? d_in[idx] : INT_MAX;
    }
    __syncthreads();

    // Bitonic sort (работает хорошо на степенях двойки)
    // Это "локальная" сортировка внутри блока.
    for (int k = 2; k <= CHUNK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < CHUNK; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    int a = s[i];
                    int b = s[ixj];

                    if ((ascending && a > b) || (!ascending && a < b)) {
                        s[i] = b;
                        s[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Выгружаем обратно (только реальные элементы, без паддинга)
    for (int i = tid; i < CHUNK; i += blockDim.x) {
        int idx = base + i;
        if (idx < n) d_out[idx] = s[i];
    }
}

// Вспомогательная функция для merge-path разбиения
// Находим такую пару (a, b), что a+b = diag, и:
// A[a-1] <= B[b] и B[b-1] < A[a] (граница раздела)
__device__ __forceinline__ int2 mergePathPartition(
    const int* A, int aCount,
    const int* B, int bCount,
    int diag
) {
    // Ограничиваем a: [max(0, diag - bCount), min(diag, aCount)]
    int aMin = max(0, diag - bCount);
    int aMax = min(diag, aCount);

    while (aMin < aMax) {
        int a = (aMin + aMax) >> 1;
        int b = diag - a;

        int Aleft  = (a > 0)     ? A[a - 1] : INT_MIN;
        int Aright = (a < aCount)? A[a]     : INT_MAX;
        int Bleft  = (b > 0)     ? B[b - 1] : INT_MIN;
        int Bright = (b < bCount)? B[b]     : INT_MAX;

        // Хотим: Aleft <= Bright и Bleft < Aright
        if (Aleft <= Bright && Bleft < Aright) {
            return make_int2(a, b);
        }

        // Если Aleft > Bright, значит a слишком большое
        if (Aleft > Bright) {
            aMax = a;
        } else {
            aMin = a + 1;
        }
    }

    int a = aMin;
    int b = diag - a;
    return make_int2(a, b);
}

// Kernel слияния двух отсортированных отрезков длиной width
// Мерджим [start .. start+width) и [start+width .. start+2*width)
// Результат пишем в d_out.
// Один блок делает одно слияние "пары".
__global__ void mergePass(
    const int* __restrict__ d_in,
    int* __restrict__ d_out,
    int n,
    int width
) {
    int pairId = blockIdx.x;            // номер пары сегментов
    int start = pairId * (2 * width);   // старт для этой пары

    if (start >= n) return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    const int* A = d_in + start;
    const int* B = d_in + mid;
    int aCount = mid - start;
    int bCount = end - mid;

    // Итоговая длина мерджа
    int total = aCount + bCount;

    // Каждый поток берет свой "диапазон диагоналей" в merge path
    int tid = threadIdx.x;
    int tcount = blockDim.x;

    int diag0 = (total * tid) / tcount;
    int diag1 = (total * (tid + 1)) / tcount;

    int2 p0 = mergePathPartition(A, aCount, B, bCount, diag0);
    int2 p1 = mergePathPartition(A, aCount, B, bCount, diag1);

    int a0 = p0.x, b0 = p0.y;
    int a1 = p1.x, b1 = p1.y;

    int outPos = start + diag0;

    // Сливаем "свой кусок" последовательно (но куски идут параллельно по потокам)
    while (a0 < a1 && b0 < b1) {
        int va = A[a0];
        int vb = B[b0];
        if (va <= vb) {
            d_out[outPos++] = va;
            a0++;
        } else {
            d_out[outPos++] = vb;
            b0++;
        }
    }
    while (a0 < a1) d_out[outPos++] = A[a0++];
    while (b0 < b1) d_out[outPos++] = B[b0++];
}

// GPU merge sort: сортируем чанки, потом сливаем чанки попарно
void gpuMergeSort(vector<int>& h, float& ms_total) {
    int n = (int)h.size();

    int* d_a = nullptr;
    int* d_b = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_a, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));

    // 1) Сортируем чанки (каждый блок - один чанк)
    int numChunks = (n + CHUNK - 1) / CHUNK;
    sortChunksBitonic<<<numChunks, THREADS>>>(d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());

    // Теперь d_b содержит отсортированные чанки
    // Будем чередовать (ping-pong) d_in и d_out
    const int* d_in = d_b;
    int* d_out = d_a;

    // 2) Итеративные проходы слияния
    int width = CHUNK;
    while (width < n) {
        int numPairs = (n + (2 * width) - 1) / (2 * width);
        mergePass<<<numPairs, THREADS>>>(d_in, d_out, n, width);
        CUDA_CHECK(cudaGetLastError());

        // меняем местами вход и выход
        const int* tmpIn = d_in;
        d_in = d_out;
        d_out = (int*)tmpIn;

        width *= 2;
    }

    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, evStart, evStop));

    // d_in сейчас указывает на актуально отсортированный массив
    CUDA_CHECK(cudaMemcpy(h.data(), d_in, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

// CPU сортировка для сравнения (обычный std::sort)
double cpuSort(vector<int> h) {
    auto t0 = chrono::high_resolution_clock::now();
    sort(h.begin(), h.end());
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;
    return ms.count();
}

// Проверка корректности: массив должен быть неубывающим
bool isSorted(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); i++) {
        if (a[i - 1] > a[i]) return false;
    }
    return true;
}

int main() {
    // По заданию нужно проверить производительность для 10000 и 100000
    vector<int> tests = {10000, 100000};

    std::mt19937 rng((unsigned)time(nullptr));
    std::uniform_int_distribution<int> dist(1, 100000);

    for (int n : tests) {
        vector<int> h(n);
        for (int i = 0; i < n; i++) h[i] = dist(rng);

        // CPU baseline
        double cpu_ms = cpuSort(h);

        // GPU sort
        vector<int> h_gpu = h;
        float gpu_ms = 0.0f;

        gpuMergeSort(h_gpu, gpu_ms);

        // Проверка
        bool ok = isSorted(h_gpu);

        cout << "N = " << n << "\n";
        cout << "CPU std::sort time: " << cpu_ms << " ms\n";
        cout << "GPU merge sort time: " << gpu_ms << " ms\n";
        cout << "Check sorted: " << (ok ? "OK" : "ERROR") << "\n";

        // "Вывод" как студент:
        if (gpu_ms < cpu_ms) {
            cout << "Conclusion: GPU version is faster on this run.\n\n";
        } else {
            cout << "Conclusion: CPU version is faster or comparable on this run (overheads may dominate).\n\n";
        }
    }

    return 0;
}