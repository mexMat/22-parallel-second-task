#define _USE_MATH_DEFINES
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "mpi.h"
#define TARGET 4.0*M_PI/24.0
int main(int argc, char *argv[]) 
{
    int i, rank, size, iter = 1, stop = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double sqr_yz, y, z, proc_sum = 0.0,
           acc_sum = 0.0, result = 0.0, eps = std::atof(argv[1]);
    int seed = 1337, block_size = 50.0/(size*eps);
    int old_seed = seed;
    block_size *= 16;
    drand48_data rand_buf;
    drand48_data rand_arr[16];
    int block_r = size*block_size/16;
    for(i = 0; i < 16; ++i)
        srand48_r(seed+i, &rand_arr[i]);
    
    double start_time = MPI_Wtime();
    while(stop){
      proc_sum = 0.0;
      for(i = 0; i < block_size; ++i)
      {
        drand48_r(&rand_arr[(i/block_r)*size + rank], &y);
        drand48_r(&rand_arr[(i/block_r)*size + rank], &z);
        sqr_yz = y*y + z*z;
        if (sqr_yz <= 1){
          proc_sum += sqrt(sqr_yz)/(size*block_size);
        }
      }
      MPI_Allreduce(&proc_sum, &acc_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      result += acc_sum;
      if (std::fabs(TARGET - result/iter) < eps){
        stop = 0;
      }
      ++iter;
    }
    double end_time = MPI_Wtime();
    
    double result_time, time = end_time - start_time;
    MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "Integral: " << 8.0*result/(iter-1) << std::endl;
      std::cout << "Eps: " << std::abs(TARGET - result/(iter-1)) << std::endl;
      std::cout << "N points: " << size * block_size * (iter-1) << std::endl;
      std::cout << "Time: " << result_time << std::endl;
      std::cout << std::endl;
    }
    MPI_Finalize();
    return 0;
}
