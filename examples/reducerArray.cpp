#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <utility>
#include <memory>
#include <cxxabi.h>

#include <RAJA/RAJA.hpp>
#include "reducerArray.hpp"

//using exec_pol = RAJA::omp_parallel_for_exec;
using exec_pol = RAJA::cuda_exec<256>;

//using red_pol = RAJA::omp_reduce_ordered;
//using red_pol = RAJA::omp_reduce;
using red_pol = RAJA::cuda_reduce_atomic<256>;

const int MAX_REGIONS = 8;


int main() {
  int N = 16;
  double* a;
  //double* a = new double[N];
  int status;
  char* realname;
  ReduceArray<RAJA::ReduceSum<red_pol, double>> reducers(MAX_REGIONS, 0.0);

  const std::type_info& r1 = typeid(RAJA::ReduceSum<red_pol,double>);
  realname = abi::__cxa_demangle(r1.name(), 0, 0, &status);
  printf("Type Reduce = %s\n",realname);
  cudaMallocManaged((void**)&a,(unsigned long)(N * sizeof(double)));
  std::iota(a, a + N, 0);
  std::transform(a, a + N, a, [=](double v) { return v / N; });
  RAJA::ReduceSum<red_pol,double> rr = reducers[0];
  RAJA::forall<exec_pol>(0, N, [=]  __device__(int i) {
  //RAJA::forall<exec_pol >(0, N, [=] (int i) {
    int reducerId = (i % MAX_REGIONS);
    reducers[reducerId] += a[i];
  });
  for (int i = 0; i < MAX_REGIONS; ++i)
    std::cout << reducers[i].get() << std::endl;

  return EXIT_SUCCESS;
}



/*
 *
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <utility>

#include <RAJA/RAJA.hpp>

using exec_pol = RAJA::omp_parallel_for_exec;
using red_pol = RAJA::omp_reduce;

int main() {
  int reducerCount;
  int N;
  std::cin >> N >> reducerCount;
  ReduceArray<RAJA::ReduceSum<red_pol, double>> reducers(reducerCount, 0.0);
  double* a = new double[N];
  std::iota(a, a + N, 0);
  std::transform(a, a + N, a, [=](double v) { return v / N; });
  RAJA::forall<exec_pol>(0, N, [=] (int i) {
    int reducerId = i % reducerCount;
    reducers[reducerId] += a[i];
  });
  for (int i = 0; i < reducerCount; ++i)
    std::cout << reducers[i].get() << std::endl;
  return EXIT_SUCCESS;
}
*/
