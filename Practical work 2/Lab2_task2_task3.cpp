#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>

using namespace std;

// 3 последовательные сортировки (из задания 1)

void bubbleSort(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

void selectionSort(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; i++) {
        int minPos = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[minPos]) minPos = j;
        }
        swap(a[i], a[minPos]);
    }
}

void insertionSort(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// Проверка корректности (на всякий случай)
bool isSorted(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); i++) {
        if (a[i - 1] > a[i]) return false;
    }
    return true;
}

// Замер времени одной сортировки
template <typename Func>
double measureSortMs(Func sortFunc, const vector<int>& base) {
    vector<int> tmp = base; // сортируем копию, чтобы сравнение было честным
    auto t0 = chrono::high_resolution_clock::now();
    sortFunc(tmp);
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;

    // минимальная проверка результата
    if (!isSorted(tmp)) {
        cerr << "ERROR: array is not sorted!\n";
    }
    return ms.count();
}

int main() {
    // 1) Размеры по заданию (пример): 1000, 10000, 100000
    vector<int> sizes = {1000, 10000, 100000};

    // 2) Генератор случайных чисел
    mt19937 rng((unsigned)time(nullptr));
    uniform_int_distribution<int> dist(1, 100000);

    cout << "OpenMP max threads = " << omp_get_max_threads() << "\n\n";

    // Здесь мы распараллеливаем внешний цикл по размерам.
    // Каждый поток работает со своим размером и своим массивом - конфликтов нет.
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)sizes.size(); idx++) {
        int n = sizes[idx];

        // Создаём исходный массив для данного размера
        vector<int> base(n);
        for (int i = 0; i < n; i++) base[i] = dist(rng);

        // Чтобы вывод не смешивался (так как печатают разные потоки),
        // собираем результаты в локальные переменные, а печать делаем в critical.
        double tBubble = 0.0, tSelect = 0.0, tInsert = 0.0;

        // Ещё один вариант распараллеливания: 3 независимые сортировки
        // (каждая сортирует копию base) можно запустить параллельно через sections.
        // Это как раз “внешние независимые задачи”.
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                tBubble = measureSortMs(bubbleSort, base);
            }
            #pragma omp section
            {
                tSelect = measureSortMs(selectionSort, base);
            }
            #pragma omp section
            {
                tInsert = measureSortMs(insertionSort, base);
            }
        }

        #pragma omp critical
        {
            cout << "N = " << n << "\n";
            cout << "  Bubble sort time:    " << tBubble << " ms\n";
            cout << "  Selection sort time: " << tSelect << " ms\n";
            cout << "  Insertion sort time: " << tInsert << " ms\n";

            cout << "  Conclusion: for larger N these O(N^2) sorts become much slower.\n\n";
        }
    }

    return 0;
}