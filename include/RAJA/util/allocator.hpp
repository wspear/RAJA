#ifndef RAJA_ALLOCATORS_HPP
#define RAJA_ALLOCATORS_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{
namespace util
{

template<>
struct allocator<Platform::host>
{
  template <typename T>
  static T *allocate(std::size_t size)
  {
    return new T[size];
  }

  template <typename T>
  static void deallocate(T* &ptr)
  {
    if (ptr) {
      delete[] ptr;
      ptr = nullptr;
    }
  }

  template <typename T>
  static T* get(T* ptr, std::size_t)
  {
    return ptr;
  }
};

#if defined(RAJA_ENABLE_CUDA)
template<>
struct allocator<Platform::cuda>
{
  template <typename T>
  static T *allocate(std::size_t size)
  {
    T* ptr;
    cudaErrchk(cudaMallocManaged(
        (void **)&ptr, 
        sizeof(T) * size, 
        cudaMemAttachGlobal));
    return ptr;
  }

  template <typename T>
  static void deallocate(T* &ptr)
  {
    if (ptr) {
      cudaErrchk(cudaFree(ptr));
      ptr = nullptr;
    }
  }

  template <typename T>
  static T* get(T* ptr, std::size_t size)
  {
    T* ret = new T[size];
    cudaErrchk(
        cudaMemcpy((void*) ret, (void*) ptr, size, cudaMemcpyDeviceToHost));
    return ret;
  }
};
#endif

#if defined(RAJA_ENABLE_OPENMP_TARGET)
template<>
struct allocator<Platform::omp_target>
{
  template <typename T>
  static T *allocate(std::size_t size)
  {
    int id = omp_get_default_device();
    T* ptr = static_cast<T>(omp_target_alloc(len * sizeof(T), id));
    return ptr;
  }

  template <typename T>
  static void deallocate(T* &ptr)
  {
    if (ptr) {
      int id = omp_get_default_device();
      omp_target_free( ptr, id );
      ptr = nullptr;
    }
  }

  template <typename T>
  static T* get(T* ptr, std::size_t size)
  {
    int hid = omp_get_initial_device();
    int did = omp_get_default_device();

    T* ret = new T[size];
    omp_target_memcpy(ret, ptr,
                      len * sizeof(T),
                      0, 0, hid, did);
    return ret;
  }
};
#endif

}
}

#endif
