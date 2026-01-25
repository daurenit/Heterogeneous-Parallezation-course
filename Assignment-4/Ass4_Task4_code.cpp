#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>

/* общая Идея:
  1) Процесс 0 формирует входной массив A размера N
  2) Делим A на куски (примерно поровну) и раздаём через MPI_Scatterv
  3) Каждый процесс локально вычисляет результат для своего куска:
       y[i] = a[i] * k + 1
     и одновременно считает локальную сумму элементов y
  4) Собираем:
     - общий массив y на процессе 0 через MPI_Gatherv (для проверки)
     - глобальную сумму через MPI_Reduce
  5) Замер времени: Scatter + compute + Gather + Reduce
*/

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int N = 1'000'000;     // размер массива
  const float k = 2.5f;        // коэффициент для обработки

  //Расчёт размеров кусков для каждого процесса (counts) и смещений (displs) 
  // Делим N элементов на size процессов: первые (N % size) процессов получают на 1 элемент больше.
  std::vector<int> counts(size, 0), displs(size, 0);
  int base = N / size;
  int rem  = N % size;

  for (int p = 0; p < size; p++) {
    counts[p] = base + (p < rem ? 1 : 0);
  }
  displs[0] = 0;
  for (int p = 1; p < size; p++) {
    displs[p] = displs[p - 1] + counts[p - 1];
  }

  //  Входной массив A формируем только на rank 0 
  std::vector<float> A;
  if (rank == 0) {
    A.resize(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    for (int i = 0; i < N; i++) A[i] = dist(rng);
  }

  //  Локальные буферы каждого процесса 
  int localN = counts[rank];
  std::vector<float> localA(localN);
  std::vector<float> localY(localN);

  // Барьер, чтобы начать замер синхронно
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  //  1) Scatter: раздать куски A всем процессам 
  MPI_Scatterv(
    rank == 0 ? A.data() : nullptr,  // sendbuf (только у rank 0)
    counts.data(),                   // sendcounts
    displs.data(),                   // displs
    MPI_FLOAT,                       // тип
    localA.data(),                   // recvbuf
    localN,                          // recvcount
    MPI_FLOAT,
    0,
    MPI_COMM_WORLD
  );

  //  2) Локальная обработка и локальная сумма 
  double localSum = 0.0;
  for (int i = 0; i < localN; i++) {
    localY[i] = localA[i] * k + 1.0f;
    localSum += localY[i];
  }

  //  3) Gather: собрать весь Y на rank 0 (для проверки) 
  std::vector<float> Y;
  if (rank == 0) Y.resize(N);

  MPI_Gatherv(
    localY.data(),
    localN,
    MPI_FLOAT,
    rank == 0 ? Y.data() : nullptr,
    counts.data(),
    displs.data(),
    MPI_FLOAT,
    0,
    MPI_COMM_WORLD
  );

  //  4) Reduce: получить глобальную сумму на rank 0 
  double globalSum = 0.0;
  MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Барьер, чтобы корректно завершить замер
  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();

  //  Проверка на rank 0 (несколько элементов) 
  if (rank == 0) {
    bool ok = true;

    // Проверяем несколько индексов: сравниваем с тем, что получилось бы формулой напрямую
    for (int idx : {0, 1, 2, 123, 999999}) {
      float ref = A[idx] * k + 1.0f;
      if (std::fabs(Y[idx] - ref) > 1e-4f) { ok = false; break; }
    }

    std::cout << "N = " << N << "\n";
    std::cout << "MPI processes = " << size << "\n";
    std::cout << "Check (few points) = " << (ok ? "OK" : "ERROR") << "\n";
    std::cout << "Global sum (Y) = " << globalSum << "\n";
    std::cout << "Total time (scatter+compute+gather+reduce) = " << (t1 - t0) * 1000.0 << " ms\n\n";
  }

  MPI_Finalize();
  return 0;
}