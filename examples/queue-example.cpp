#include "RAJA/RAJA.hpp"

#include <iostream>


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  /*
   * Pointer so they can be captured by the enqueued lambdas....
   */
  RAJA::Queue* q1 = new RAJA::Queue();
  RAJA::Queue* q2 = new RAJA::Queue();

  double* a = new double[10];

  double* b = new double[10];

  double c = 3.14;

  q1->enqueue( [=] (auto context) {
      RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0,10), [=] (int i) {
          a[i] = i;
      });
      std::cout << "a[5] = " << a[5] << std::endl;

  q2->enqueue( [=] (auto context) {
      RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0,10), [=] (int i) {
          b[i] = i;
      });
  });

  q2->enqueue( [=] (auto context) {
      q1->wait();

      RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0,10), [=] (int i) {
          a[i] += b[i]*c;
      });

  });

  q2->wait();

  std::cout << "a[5] = " << a[5] << std::endl;
}
