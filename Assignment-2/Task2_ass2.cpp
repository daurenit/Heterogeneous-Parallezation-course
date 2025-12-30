#include <iostream>
#include <vector>
#include <cstdlib>   // rand, srand
#include <ctime>     // time
#include <chrono>    // замер времени
#include <omp.h>     // OpenMP

using namespace std;

int main() {
    // По заданию: массив из 10 000 случайных чисел
    const int N = 10000;

    // Создаём массив (vector проще и безопаснее, чем new/delete)
    vector<int> a(N);

    // Инициализируем генератор случайных чисел один раз
    srand((unsigned)time(nullptr));

    // Заполняем массив числами (пусть будет 1..100000, чтобы min/max было интереснее)
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100000 + 1;
    }

    // 1) Последовательный поиск min и max + замер времени
    auto t1 = chrono::high_resolution_clock::now();

    int min_seq = a[0];
    int max_seq = a[0];

    // Обычный проход по массиву: обновляем min/max
    for (int i = 1; i < N; i++) {
        if (a[i] < min_seq) min_seq = a[i];
        if (a[i] > max_seq) max_seq = a[i];
    }

    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time = t2 - t1;

    // 2) Параллельный поиск min и max (OpenMP) + замер времени
    auto t3 = chrono::high_resolution_clock::now();

    // Важно: если несколько потоков будут писать в один min/max — будет гонка данных.
    // Поэтому делаем "локальные" min/max для каждого потока, а потом объединяем.
    int min_par = a[0];
    int max_par = a[0];

    #pragma omp parallel
    {
        int local_min = a[0];
        int local_max = a[0];

        // Каждый поток обрабатывает свою часть индексов
        #pragma omp for nowait
        for (int i = 1; i < N; i++) {
            if (a[i] < local_min) local_min = a[i];
            if (a[i] > local_max) local_max = a[i];
        }

        // Сводим результат: сюда потоки заходят по очереди (critical),
        // чтобы безопасно обновить общие min_par/max_par
        #pragma omp critical
        {
            if (local_min < min_par) min_par = local_min;
            if (local_max > max_par) max_par = local_max;
        }
    }

    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time = t4 - t3;

    // Вывод результатов
    cout << "N = " << N << "\n\n";

    cout << "Sequential:\n";
    cout << "  min = " << min_seq << ", max = " << max_seq << "\n";
    cout << "  time = " << seq_time.count() << " sec\n\n";

    cout << "Parallel (OpenMP):\n";
    cout << "  min = " << min_par << ", max = " << max_par << "\n";
    cout << "  time = " << par_time.count() << " sec\n\n";

    // Простая проверка, что результаты совпали
    if (min_seq == min_par && max_seq == max_par) {
        cout << "Check: OK (results match)\n";
    } else {
        cout << "Check: ERROR (results do not match)\n";
    }

    // Формулируем выводы:
    if (par_time.count() < seq_time.count()) {
        cout << "Conclusion: parallel version is faster on this run.\n";
    } else {
        cout << "Conclusion: sequential version is faster or equal on this run (overheads may dominate).\n";
    }

    return 0;
}