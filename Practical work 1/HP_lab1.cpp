#include <iostream>
#include <cstdlib>   // для rand, srand
#include <ctime>     // для вычисления time
#include <omp.h>     // OpenMP

using namespace std;

// функция для вычисления среднего (просто последовательно)
double averageSequential(const int* arr, int n) {
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return (double)sum / n;
}

// функция для вычисления среднего (параллельно через OpenMP reduction)
double averageParallel(const int* arr, int n) {
    long long sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }

    return (double)sum / n;
}

int main() {
    int n;
    cout << "Enter array size: ";
    cin >> n;

    // Динамически выделяем память под массив
    int* arr = new int[n];

    // Инициализируем генератор случайных чисел
    srand((unsigned)time(nullptr));

    // Заполняем массив случайными числами
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100 + 1;
    }

    // Последовательное среднее
    double start_seq = omp_get_wtime();
    double avg_seq = averageSequential(arr, n);
    double end_seq = omp_get_wtime();

    // Параллельное среднее
    double start_par = omp_get_wtime();
    double avg_par = averageParallel(arr, n);
    double end_par = omp_get_wtime();

    // Вывод результатов
    cout << "\nSequential average: " << avg_seq << endl;
    cout << "Parallel average:   " << avg_par << endl;

    cout << "\nSequential time: " << (end_seq - start_seq) << " sec" << endl;
    cout << "Parallel time:   " << (end_par - start_par) << " sec" << endl;

    // Освобождаем память
    delete[] arr;

    return 0;
}