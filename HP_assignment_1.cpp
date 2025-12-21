
// // Task 1 откоменьте чтобы проверить

// #include <iostream> // для std::cout, std::endl
// #include <cstdlib>  // для std::rand, std::srand
// #include <ctime>    // для std::time

// int main() {
//     const int N = 50000;         // Размер массива 50 000 целых чисел

//     // Динамическое выделение памяти под массив
//     int* arr = new int[N];

//     // Инициализация генератора случайных чисел
//     std::srand(static_cast<unsigned>(std::time(nullptr)));

//     // Заполнение массива случайными числами от 1 до 100
//     for (int i = 0; i < N; ++i) {
//         arr[i] = std::rand() % 100 + 1;
//     }

//     // Вычисление суммы и среднего значения
//     long long sum = 0;
//     for (int i = 0; i < N; ++i) {
//         sum += arr[i];
//     }

//     double average = static_cast<double>(sum) / N;

//     // Вывод результата
//     std::cout << "Average value of array elements: " << average << std::endl;

//     // Корректное освобождение памяти
//     delete[] arr;

//     return 0;
// }



// // Task 2 откоментьте чтобы проверить

// #include <iostream>
// #include <cstdlib>     // для rand(), srand()
// #include <ctime>       // для time()
// #include <chrono>      // для измерение времени

// using namespace std;

// int main() {

//     const int N = 1000000; // размер массива

//     // Выделяем память динамически
//     int* arr = new int[N];

//     // Готовим генератор случайных чисел
//     srand(time(nullptr));

//     // Заполняем массив случайными числами
//     for (int i = 0; i < N; i++) {
//         arr[i] = rand() % 100000;
//     }

//     // Замеряем время поиска min и max
//     auto start = chrono::high_resolution_clock::now();

//     int minVal = arr[0];
//     int maxVal = arr[0];

//     for (int i = 1; i < N; i++) {
//         if (arr[i] < minVal) minVal = arr[i];
//         if (arr[i] > maxVal) maxVal = arr[i];
//     }

//     auto end = chrono::high_resolution_clock::now();

//     chrono::duration<double> duration = end - start;

//     // Вывод результатов
//     cout << "Min value: " << minVal << endl;
//     cout << "Max value: " << maxVal << endl;
//     cout << "Sequential time: " << duration.count() << " sec" << endl;

//     // Освобождаем память
//     delete[] arr;

//     return 0;
// }

// // Task 3 - откоментьте чтобы проверить

// #include <iostream>
// #include <cstdlib>      // для rand(), srand()
// #include <ctime>        // для time()
// #include <chrono>       // для измерение времени std::chrono
// #include <climits>      // для INT_MAX, INT_MIN
// #include <omp.h>        // для OpenMP

// using namespace std;

// int main() {

//     const int N = 1000000; // размер массива

//     // Динамическое выделение памяти
//     int* arr = new int[N];

//     // Инициализация генератора случайных чисел
//     srand(static_cast<unsigned>(time(nullptr)));

//     // Заполнение массива случайными числами
//     for (int i = 0; i < N; i++) {
//         arr[i] = rand() % 100000; // диапазон условный
//     }

//     // Последовательный поиск min и max
//     auto start_seq = chrono::high_resolution_clock::now();

//     int min_seq = arr[0];
//     int max_seq = arr[0];

//     for (int i = 1; i < N; i++) {
//         if (arr[i] < min_seq) min_seq = arr[i];
//         if (arr[i] > max_seq) max_seq = arr[i];
//     }

//     auto end_seq = chrono::high_resolution_clock::now();
//     chrono::duration<double> time_seq = end_seq - start_seq;

//     // Параллельный поиск min и max (OpenMP + reduction)
//     auto start_par = chrono::high_resolution_clock::now();

//     int min_par = INT_MAX;   // начальные значения для редукции
//     int max_par = INT_MIN;

//     #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
//     for (int i = 0; i < N; i++) {
//         if (arr[i] < min_par) min_par = arr[i];
//         if (arr[i] > max_par) max_par = arr[i];
//     }

//     auto end_par = chrono::high_resolution_clock::now();
//     chrono::duration<double> time_par = end_par - start_par;

//     // Вывод результатов
//     cout << "Sequential (single thread):" << endl;
//     cout << "  min = " << min_seq << ", max = " << max_seq << endl;
//     cout << "  time = " << time_seq.count() << " sec" << endl;

//     cout << "\nParallel (OpenMP):" << endl;
//     cout << "  min = " << min_par << ", max = " << max_par << endl;
//     cout << "  time = " << time_par.count() << " sec" << endl;

//     // Проверка корректности
//     if (min_seq != min_par || max_seq != max_par) {
//         cerr << "\nWARNING: results are different! (something is wrong)\n";
//     }

//     // Освобождение памяти
//     delete[] arr;

//     return 0;
// }



// // Task 4 - откоменьте чтобы проверить

// #include <iostream>
// #include <cstdlib>      // для rand(), srand()
// #include <ctime>        // для time()
// #include <chrono>       // для измерение времени
// #include <omp.h>        // для OpenMP

// using namespace std;

// int main() {
//     const int N = 5000000;

//     // Динамически выделяем память под массив
//     int* arr = new int[N];

//     // Инициализируем генератор случайных чисел
//     srand(static_cast<unsigned>(time(nullptr)));

//     // Заполняем массив случайными числами от 1 до 100
//     for (int i = 0; i < N; ++i) {
//         arr[i] = rand() % 100 + 1;
//     }

//     // Последовательное вычисление среднего значения
//     auto start_seq = chrono::high_resolution_clock::now();

//     long long seq_sum = 0;         // сумма в последовательной версии
//     for (int i = 0; i < N; ++i) {
//         seq_sum += arr[i];
//     }

//     double seq_avg = static_cast<double>(seq_sum) / N;

//     auto end_seq = chrono::high_resolution_clock::now();
//     chrono::duration<double> time_seq = end_seq - start_seq;

//     // Параллельное вычисление среднего с OpenMP + reduction
//     auto start_par = chrono::high_resolution_clock::now();

//     long long par_sum = 0;         // общая сумма для параллельной версии

//     #pragma omp parallel for reduction(+:par_sum)
//     for (int i = 0; i < N; ++i) {
//         par_sum += arr[i];
//     }

//     double par_avg = static_cast<double>(par_sum) / N;

//     auto end_par = chrono::high_resolution_clock::now();
//     chrono::duration<double> time_par = end_par - start_par;

//     // Вывод результатов и сравнение
//     cout << "Sequential average: " << seq_avg << endl;
//     cout << "Parallel   average: " << par_avg << endl;

//     cout << "\nSequential time: " << time_seq.count() << " sec\n";
//     cout << "Parallel   time: " << time_par.count() << " sec\n";

//     if (abs(seq_avg - par_avg) > 1e-9) {
//         cerr << "\nWARNING: averages are different! (что-то не так)\n";
//     }

//     // Освобождаем память
//     delete[] arr;

//     return 0;
// }