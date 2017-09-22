#include <memory>
#include <RAJA/RAJA.hpp>
#include <RAJA/policy/cuda/cudaUM.hpp>


#if 1 // active prototype

#include <memory>

template <typename ReduceType>
class ReduceArray {
  //constexpr ReduceType* alloc(unsigned int count) noexcept {
  ReduceType* alloc(unsigned int count) noexcept {
    return static_cast<ReduceType*>(::operator new(sizeof(ReduceType) * count));
  }

public:

  template <typename T>
  explicit ReduceArray(unsigned int count, T value) noexcept
  : data(alloc(count+1)), size(count) {
    // we allocate one more than requested as an auxilliary needed to satify some logic with parent pointers
    for (unsigned int i = 0; i <= size; ++i)
      new (std::addressof(data[i])) ReduceType(value);
#if defined(RAJA_ENABLE_CUDA)
    cudaMalloc((void**)&dataDevice,(size+1) * sizeof(ReduceType));
    cudaMemcpy(dataDevice,data, (size+1) * sizeof(ReduceType), cudaMemcpyHostToDevice);
    printf("ReduceArray Host constructor Addresses %p %p : data %p dataDevice %p\n",data,dataDevice,&data[0],&dataDevice[0]);
    
#endif    
  }

  RAJA_HOST_DEVICE
  ReduceArray(ReduceArray const& other) noexcept
#if !defined(__CUDA_ARCH__)
  //: data(alloc(other.size)), size(other.size) 
  {
  //  for (unsigned int i = 0; i < size; ++i)
  //    new (std::addressof(data[i])) ReduceType(other.data[i]);
    data = other.data;
    dataDevice = other.dataDevice;
    size = other.size;
    bool deviceSetup = false;
    printf("ReduceArray Host copy constructor : data %p dataDevice %p\n",&data[0],&dataDevice[0]);
    for (unsigned int i = 0; i < size; ++i) {
      deviceSetup = data[i].auxSetup();
    }  
#if defined(RAJA_ENABLE_CUDA)
    if(deviceSetup) {
      printf("deviceSetup Ready calling cudaMemcpy\n");
      cudaMemcpy(dataDevice,data, (size+1) * sizeof(ReduceType), cudaMemcpyHostToDevice);
    }
#endif    
  }
#else
  {
    data = other.data;
    dataDevice = other.dataDevice;
    size = other.size;
    dataDevice[size].setParent(nullptr);
    for (unsigned int i = 0; i < size; ++i) {
      dataDevice[i].setParent(&dataDevice[size]);
    }  
    //printf("ReduceArray Device copy constructor : data %p dataDevice %p\n",&data[0],&dataDevice[0]);
  }
#endif 


  RAJA_HOST_DEVICE
  ~ReduceArray() noexcept {
#if !defined(__CUDA_ARCH__)
    //for (unsigned int i = 0; i < size; ++i)
    //  data[i].~ReduceType();
    printf("~ReduceArray Host\n");
#else
    //printf("~ReduceArray Device\n");
    for (unsigned int i = 0; i < size; ++i)
      dataDevice[i].~ReduceType();

#endif    
  }

  RAJA_HOST_DEVICE
  ReduceType& operator[] (unsigned int i) const noexcept {
#if !defined(__CUDA_ARCH__)
    //printf("operator[] on host with [%d] at address %p\n",i,&data[i]);
    return data[i];
#else
    //printf("operator[] on device with [%d] at address %p  and %p \n",i,&data[i],&dataDevice[i]);
    return dataDevice[i];
#endif    
  }

//private:
  struct Deleter {
    void operator()(ReduceType* p) const noexcept {
      delete p;
    }
  };

  
  //std::unique_ptr<ReduceType[], Deleter> data;
  ReduceType* data;
  ReduceType* dataDevice;
  unsigned int size;
};


#endif



