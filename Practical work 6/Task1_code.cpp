#include <OpenCL/opencl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>

// Чтение OpenCL-ядра из файла
std::string loadKernel(const std::string& filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    const int N = 1'000'000;
    const size_t bytes = N * sizeof(float);

    // --- Данные на CPU ---
    std::vector<float> A(N), B(N), C(N), C_ref(N);
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
        C_ref[i] = A[i] + B[i];
    }

    // --- Получение платформы ---
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // --- Получение устройства (GPU, если есть) ---
    cl_device_id device;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

if (err != CL_SUCCESS) {
    // Если GPU недоступен — используем CPU
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
}

    // --- Контекст и очередь ---
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // --- Буферы ---
    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               bytes, A.data(), nullptr);
    cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               bytes, B.data(), nullptr);
    cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                               bytes, nullptr, nullptr);

    // --- Загрузка и сборка ядра ---
    std::string source = loadKernel("Task1_kernel.cl");
    const char* src = source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "vector_add", nullptr);

    // --- Аргументы ядра ---
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    // --- Настройка NDRange ---
    size_t localSize = 256;
    size_t globalSize = ((N + localSize - 1) / localSize) * localSize;

    // --- Запуск и замер времени ---
    auto t1 = std::chrono::high_resolution_clock::now();

    clEnqueueNDRangeKernel(queue, kernel, 1,
                           nullptr, &globalSize, &localSize,
                           0, nullptr, nullptr);

    clFinish(queue);

    auto t2 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // --- Копирование результата ---
    clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr);

    // --- Проверка ---
    bool ok = true;
    for (int i : {0, 1, 2, 123, 999999}) {
        if (std::fabs(C[i] - C_ref[i]) > 1e-5f) {
            ok = false;
            break;
        }
    }

    std::cout << "N = " << N << "\n";
    std::cout << "Check = " << (ok ? "OK" : "ERROR") << "\n";
    std::cout << "OpenCL time = " << gpu_ms << " ms\n";

    // --- Освобождение ресурсов ---
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}