#include "RAJA/RAJA.hpp"

constexpr std::size_t SIZE = 200000000;

int main(int, char**)
{

  double* a = new double[SIZE];
  double* b = new double[SIZE];
  double* c = new double[SIZE];

  for (std::ptrdiff_t i = 0; i < SIZE; ++i) {
    a[i] = 2.0 * i;
    b[i] = 1.0 * i;
    c[i] = 0.0;
  }

  const RAJA::RangeSegment range{0, SIZE};

  RAJA::forall<RAJA::loop_exec>(range, [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });

  RAJA::kernel<
    RAJA::KernelPolicy<
      RAJA::statement::For<
#if defined(RAJA_ENABLE_OPENMP)
        0, RAJA::omp_parallel_for_exec, RAJA::statement::Lambda<0>
#else
        0, RAJA::loop_exec, RAJA::statement::Lambda<0>
#endif
      >
    >
  >(RAJA::make_tuple(range),
      [=] (int i) { 
        c[i] = a[i] + b[i]; 
  });
}
