#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <limits>
#include <omp.h>

using namespace std;

// Генерация массива случайных int
vector<int> makeRandomArray(int n, int low = 1, int high = 100000) {
    // фиксируем seed, чтобы сравнение было честным (одни и те же данные)
    static std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(low, high);

    vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = dist(rng);
    return a;
}

// Последовательная сортировка выбором
void selectionSortSequential(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        int minIdx = i;
        for (int j = i + 1; j < n; ++j) {
            if (a[j] < a[minIdx]) minIdx = j;
        }
        if (minIdx != i) std::swap(a[i], a[minIdx]);
    }
}

// Параллельная версия selection sort через OpenMP:
// Идея: на каждом шаге i нужно найти индекс минимума на отрезке
// Сам поиск минимума можно распараллелить: каждый поток ищет локальный минимум
// потом выбираем глобальный минимум и делаем swap (swap делаем один раз, последовательно)
void selectionSortOpenMP(vector<int>& a) {
    int n = (int)a.size();

    for (int i = 0; i < n - 1; ++i) {

        int globalMinVal = a[i];
        int globalMinIdx = i;

        // Каждый поток будет держать свой локальный минимум
        #pragma omp parallel
        {
            int localMinVal = globalMinVal;
            int localMinIdx = globalMinIdx;

            // Делим диапазон j между потоками
            #pragma omp for nowait
            for (int j = i + 1; j < n; ++j) {
                if (a[j] < localMinVal) {
                    localMinVal = a[j];
                    localMinIdx = j;
                }
            }

            // Сводим локальные минимумы в один глобальный (критическая секция)
            #pragma omp critical
            {
                if (localMinVal < globalMinVal) {
                    globalMinVal = localMinVal;
                    globalMinIdx = localMinIdx;
                }
            }
        }

        // После параллельного поиска - один swap
        if (globalMinIdx != i) std::swap(a[i], a[globalMinIdx]);
    }
}

bool isSortedNonDecreasing(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i] < a[i - 1]) return false;
    }
    return true;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // По заданию - проверить производительность для 1000 и 10000
    vector<int> sizes = {1000, 10000};

    cout << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";

    for (int n : sizes) {
        cout << "========== N = " << n << " ==========\n";

        // 1) Генерируем исходный массив
        vector<int> base = makeRandomArray(n);

        // 2) Sequential
        vector<int> aSeq = base;
        auto t1 = chrono::high_resolution_clock::now();
        selectionSortSequential(aSeq);
        auto t2 = chrono::high_resolution_clock::now();
        chrono::duration<double> seqTime = t2 - t1;

        // 3) OpenMP
        vector<int> aPar = base;
        auto t3 = chrono::high_resolution_clock::now();
        selectionSortOpenMP(aPar);
        auto t4 = chrono::high_resolution_clock::now();
        chrono::duration<double> parTime = t4 - t3;

        // 4) Проверка корректности
        bool ok1 = isSortedNonDecreasing(aSeq);
        bool ok2 = isSortedNonDecreasing(aPar);
        bool same = (aSeq == aPar); // для selection sort результат должен совпасть

        cout << "Sequential time: " << seqTime.count() << " sec\n";
        cout << "OpenMP time:     " << parTime.count() << " sec\n";

        if (parTime.count() > 0.0) {
            cout << "Speedup (seq/par): " << (seqTime.count() / parTime.count()) << "x\n";
        }

        cout << "Check sorted: seq=" << (ok1 ? "OK" : "ERROR")
             << ", omp=" << (ok2 ? "OK" : "ERROR") << "\n";
        cout << "Check same result: " << (same ? "OK" : "ERROR") << "\n";

        // 5) Вывод по сути задания
        if (parTime < seqTime) {
            cout << "Conclusion: OpenMP version is faster on this run.\n";
        } else {
            cout << "Conclusion: Sequential is faster or equal on this run (overheads may dominate).\n";
        }

        cout << "\n";
    }

    return 0;
}