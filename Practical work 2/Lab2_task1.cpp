#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Сортировка пузырьком
void bubbleSort(vector<int>& a) {
    int n = a.size();

    for (int i = 0; i < n - 1; i++) {
        // После каждого прохода самый большой элемент "всплывает" в конец
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// Сортировка выбором
void selectionSort(vector<int>& a) {
    int n = a.size();

    for (int i = 0; i < n - 1; i++) {
        // Считаем, что минимальный элемент находится в позиции i
        int minIndex = i;

        // Ищем минимальный элемент в неотсортированной части
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[minIndex]) {
                minIndex = j;
            }
        }

        // Ставим минимальный элемент на позицию i
        swap(a[i], a[minIndex]);
    }
}

// Сортировка вставкой
void insertionSort(vector<int>& a) {
    int n = a.size();

    for (int i = 1; i < n; i++) {
        int key = a[i];   // текущий элемент
        int j = i - 1;

        // Сдвигаем элементы вправо, пока не найдём место для key
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }

        // Вставляем элемент на нужную позицию
        a[j + 1] = key;
    }
}

// Вспомогательная функция печати массива
void printArray(const vector<int>& a) {
    for (int x : a) {
        cout << x << " ";
    }
    cout << endl;
}

int main() {
    srand(time(nullptr));

    const int N = 10; // небольшой размер для наглядности
    vector<int> arr(N);

    // Заполняем массив случайными числами
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100;
    }

    cout << "Original array:\n";
    printArray(arr);

    // Пузырёк
    vector<int> a1 = arr;
    bubbleSort(a1);
    cout << "\nBubble sort:\n";
    printArray(a1);

    // Выбор
    vector<int> a2 = arr;
    selectionSort(a2);
    cout << "\nSelection sort:\n";
    printArray(a2);

    // Вставка
    vector<int> a3 = arr;
    insertionSort(a3);
    cout << "\nInsertion sort:\n";
    printArray(a3);

    return 0;
}