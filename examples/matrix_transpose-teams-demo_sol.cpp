//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

/*
 *  Matrix Transpose Example
 *
 *  In this example, an input matrix A of dimension N_r x N_c is
 *  transposed and returned as a second matrix At of size N_c x N_r.
 *
 *  This operation is carried out using a local memory tiling
 *  algorithm. The algorithm first loads matrix entries into an
 *  iteraion shared tile, a two-dimensional array, and then
 *  reads from the tile with row and column indices swapped for
 *  the output matrix.
 *
 *  The algorithm is expressed as a collection of ``outer``
 *  and ``inner`` for loops. Iterations of the inner loops will load/read
 *  data into the tile; while outer loops will iterate over the number
 *  of tiles needed to carry out the transpose.
 *
 */

//
// Define dimensionality of matrices
//
const int DIM = 2;

//
// Function for checking results
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c);

//
// Function for printing results
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c);


int main(int argc, char *argv[])
{

  std::cout << "\n\nRAJA shared matrix transpose example...\n";

  //
  // Define num rows/cols in matrix, tile dimensions, and number of tiles
  //
  // _mattranspose_localarray_dims_start
  const int N_r = 267;
  const int N_c = 251;

  const int TILE_DIM = 16;

  const int outer_Dimc = (N_c - 1) / TILE_DIM + 1;
  const int outer_Dimr = (N_r - 1) / TILE_DIM + 1;
  // _mattranspose_localarray_dims_end

  //
  // Allocate matrix data
  //
  int *A = memoryManager::allocate<int>(N_r * N_c);
  int *At = memoryManager::allocate<int>(N_r * N_c);

  //
  // In the following implementations of matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  // _mattranspose_localarray_views_start
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_r, N_c);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_c, N_r);
  // _mattranspose_localarray_views_end

  //
  // Initialize matrix data
  //
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      Aview(row, col) = col;
    }
  }
  // printResult<int>(Aview, N_r, N_c);

  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of shared matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  // _mattranspose_localarray_cstyle_start
  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dimr; ++by) {
    for (int bx = 0; bx < outer_Dimc; ++bx) {

      // Stack-allocated local array for data on a tile
      int Tile[TILE_DIM][TILE_DIM];

      //
      // (1) Inner loops to read input matrix tile data into the array
      //
      //     Note: loops are ordered so that input matrix data access
      //           is stride-1.
      //
      for (int ty = 0; ty < TILE_DIM; ++ty) {
        for (int tx = 0; tx < TILE_DIM; ++tx) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Tile[ty][tx] = Aview(row, col);
          }
        }
      }

      //
      // (2) Inner loops to write array data into output array tile
      //
      //     Note: loop order is swapped from above so that output matrix
      //           data access is stride-1.
      //
      for (int tx = 0; tx < TILE_DIM; ++tx) {
        for (int ty = 0; ty < TILE_DIM; ++ty) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Atview(col, row) = Tile[ty][tx];
          }
        }
      }

    }
  }
  // _mattranspose_localarray_cstyle_end

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

  //----------------------------------------------------------------------------//

  std::memset(At, 0, N_r * N_c * sizeof(int));


  //Step 1. Define Launch policy
  using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t
                                                 ,RAJA::expt::cuda_launch_t<false>>;


  //Step 2. Define team and thread policies
  // RAJA CUDA/HIP _direct policies assume iteration spaces will not exceed
  // number of teams, threads
  // RAJA CUDA/HIP _loop policies use a strided loop to enables iteration spaces beyond
  // number of teams and threads.
  using teams_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                         ,RAJA::cuda_block_x_direct>;

  using teams_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                         ,RAJA::cuda_block_y_direct>;

  using threads_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                           ,RAJA::cuda_thread_x_direct>;

  using threads_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                           ,RAJA::cuda_thread_y_direct>;


  //Command line argument 0 denotes host, 1 denotes device
  RAJA::expt::ExecPlace select_cpu_or_gpu = (RAJA::expt::ExecPlace)atoi(argv[1]);

  if(select_cpu_or_gpu == RAJA::expt::HOST) {
    printf("\n Running on the host \n");
  }

  if(select_cpu_or_gpu == RAJA::expt::DEVICE) {
    printf("\n Running on the device \n");
  }

  RAJA::expt::launch<launch_policy>(select_cpu_or_gpu,
  RAJA::expt::Resources(RAJA::expt::Teams(outer_Dimc, outer_Dimr),
                        RAJA::expt::Threads(TILE_DIM, TILE_DIM)),
       [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

  //Loop methods have the following signature
  //RAJA::expt::loop<>(ctx, RAJA::RangeSegment(st,end), [&](int i) {});

  //
  // (0) Outer loops to iterate over tiles
  //
  RAJA::expt::loop<teams_y>(ctx, RAJA::RangeSegment(0, outer_Dimr), [&](int by) {
      RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, outer_Dimc), [&](int bx) {

      // Stack-allocated local array for data on a tile
      RAJA_TEAM_SHARED int Tile[TILE_DIM][TILE_DIM];

      //
      // (1) Inner loops to read input matrix tile data into the array
      //
      //     Note: loops are ordered so that input matrix data access
      //           is stride-1.
      //
      RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&](int ty) {
          RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&](int tx) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Tile[ty][tx] = Aview(row, col);
          }

        });
      });
      
       ctx.teamSync();
      //
      // (2) Inner loops to write array data into output array tile
      //
      //     Note: loop order is swapped from above so that output matrix
      //           data access is stride-1.
      //
      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&](int tx) {
       RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&](int ty) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Atview(col, row) = Tile[ty][tx];
          }
        });
       });

    });
  });

  });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);


 //--------------------------------------------------------------------------//

  return 0;
}


//
// Function to check result and report P/F.
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c)
{
  bool match = true;
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      if (Atview(row, col) != row) {
        match = false;
      }
    }
  }
  if (match) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Function to print result.
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c)
{
  std::cout << std::endl;
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
                << std::endl;
    }
    std::cout << "" << std::endl;
  }
  std::cout << std::endl;
}
