
#include <OpenCL/opencl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>

static void checkCLError(cl_int err, const char* msg) {
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL error (" << err << "): " << msg << "\n";
    std::exit(1);
  }
}

static std::string loadKernel(const std::string& fileName) {
  std::ifstream file(fileName);
  if (!file) {
    std::cerr << "Cannot open kernel file: " << fileName << "\n";
    std::exit(1);
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

static void cpu_matmul(const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C,
                       int N, int M, int K) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      float sum = 0.0f;
      for (int t = 0; t < M; t++) {
        sum += A[i * M + t] * B[t * K + j];
      }
      C[i * K + j] = sum;
    }
  }
}

static bool check_some(const std::vector<float>& ref,
                       const std::vector<float>& got,
                       int N, int K) {
  auto eq = [&](int i, int j) {
    float a = ref[i * K + j];
    float b = got[i * K + j];
    return std::fabs(a - b) <= 1e-3f;
  };

  // Несколько точек + пара случайных
  if (!eq(0, 0)) return false;
  if (!eq(N - 1, K - 1)) return false;
  if (!eq(N / 2, K / 2)) return false;

  for (int t = 0; t < 5; t++) {
    int i = (t * 37) % N;
    int j = (t * 91) % K;
    if (!eq(i, j)) return false;
  }
  return true;
}

int main() {
  // Размеры матриц: A(NxM), B(MxK), C(NxK)
  const int N = 256;
  const int M = 256;
  const int K = 256;

  const size_t bytesA = (size_t)N * M * sizeof(float);
  const size_t bytesB = (size_t)M * K * sizeof(float);
  const size_t bytesC = (size_t)N * K * sizeof(float);

  std::vector<float> hA((size_t)N * M);
  std::vector<float> hB((size_t)M * K);
  std::vector<float> hC((size_t)N * K, 0.0f);
  std::vector<float> hRef((size_t)N * K, 0.0f);

  // Инициализация данных
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto& x : hA) x = dist(rng);
  for (auto& x : hB) x = dist(rng);

  // CPU reference
  auto c1 = std::chrono::high_resolution_clock::now();
  cpu_matmul(hA, hB, hRef, N, M, K);
  auto c2 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(c2 - c1).count();

  // --- OpenCL init ---
  cl_int err = CL_SUCCESS;

  cl_uint numPlatforms = 0;
  checkCLError(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs(count)");
  if (numPlatforms == 0) {
    std::cerr << "No OpenCL platforms found.\n";
    return 1;
  }
  std::vector<cl_platform_id> platforms(numPlatforms);
  checkCLError(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");

  // Выбираем первую платформу с GPU, иначе CPU
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;

  for (auto p : platforms) {
    cl_uint numDevs = 0;
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, &numDevs) == CL_SUCCESS && numDevs > 0) {
      platform = p;
      break;
    }
  }
  if (!platform) {
    platform = platforms[0];
    cl_uint numDevs = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &numDevs);
    checkCLError(err, "clGetDeviceIDs(CPU)");
  }

  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  checkCLError(err, "clCreateContext");

  // Командная очередь (без профилирования, время меряем по host+finish)
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  checkCLError(err, "clCreateCommandQueue");

  // --- Program / Kernel ---
  std::string src = loadKernel("Task2_kernel.cl");
  const char* srcPtr = src.c_str();
  size_t srcLen = src.size();

  cl_program program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
  checkCLError(err, "clCreateProgramWithSource");

  err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::vector<char> log(logSize);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    std::cerr << "Build log:\n" << log.data() << "\n";
    checkCLError(err, "clBuildProgram");
  }

  cl_kernel kernel = clCreateKernel(program, "matmul_basic", &err);
  checkCLError(err, "clCreateKernel(matmul_basic)");

  // --- Buffers ---
  cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytesA, nullptr, &err);
  checkCLError(err, "clCreateBuffer(A)");
  cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytesB, nullptr, &err);
  checkCLError(err, "clCreateBuffer(B)");
  cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytesC, nullptr, &err);
  checkCLError(err, "clCreateBuffer(C)");

  // H2D copy
  checkCLError(clEnqueueWriteBuffer(queue, dA, CL_TRUE, 0, bytesA, hA.data(), 0, nullptr, nullptr),
               "clEnqueueWriteBuffer(A)");
  checkCLError(clEnqueueWriteBuffer(queue, dB, CL_TRUE, 0, bytesB, hB.data(), 0, nullptr, nullptr),
               "clEnqueueWriteBuffer(B)");

  // --- Kernel args ---
  checkCLError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA), "clSetKernelArg(0)");
  checkCLError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB), "clSetKernelArg(1)");
  checkCLError(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC), "clSetKernelArg(2)");
  checkCLError(clSetKernelArg(kernel, 3, sizeof(int), &N), "clSetKernelArg(3)");
  checkCLError(clSetKernelArg(kernel, 4, sizeof(int), &M), "clSetKernelArg(4)");
  checkCLError(clSetKernelArg(kernel, 5, sizeof(int), &K), "clSetKernelArg(5)");

  // --- NDRange ---
  // global: (N, K) -> каждый work-item считает один элемент C[row,col]
  // local: (16, 16) как дефолт; если устройство не поддержит — можно уменьшить.
  const size_t local[2]  = {16, 16};

  auto roundUp = [](size_t x, size_t m) {
    return ((x + m - 1) / m) * m;
  };

  const size_t global[2] = {
    roundUp((size_t)N, local[0]),
    roundUp((size_t)K, local[1])
  };

  // --- Run + timing ---
  auto t1 = std::chrono::high_resolution_clock::now();

  checkCLError(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr),
               "clEnqueueNDRangeKernel");

  // важный момент: ждём завершения
  checkCLError(clFinish(queue), "clFinish");

  auto t2 = std::chrono::high_resolution_clock::now();
  double ocl_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

  // D2H
  checkCLError(clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytesC, hC.data(), 0, nullptr, nullptr),
               "clEnqueueReadBuffer(C)");

  bool ok = check_some(hRef, hC, N, K);

  std::cout << "A: " << N << "x" << M << ", B: " << M << "x" << K << ", C: " << N << "x" << K << "\n";
  std::cout << "Check = " << (ok ? "OK" : "ERROR") << "\n";
  std::cout << "CPU time = " << cpu_ms << " ms\n";
  std::cout << "OpenCL time (kernel+finish) = " << ocl_ms << " ms\n";

  // --- Cleanup ---
  clReleaseMemObject(dA);
  clReleaseMemObject(dB);
  clReleaseMemObject(dC);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}