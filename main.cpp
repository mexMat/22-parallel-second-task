#define _USE_MATH_DEFINES
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "mpi.h"
#define TARGET 4.0*M_PI/3.0
int main(int argc, char *argv[]) {
    int i, rank, size, iter = 1, stop = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double sqr_yz, y, z, proc_sum = 0.0, acc_sum = 0.0, result = 0.0, eps = std::atof(argv[1]);
    int block_size = 1000/size;
    srand(rank + 1);
    double start_time = MPI_Wtime();
    while(stop){
        for(i = 0; i < block_size; ++i)
        {
            y = (double)rand()/RAND_MAX;
            z = (double)rand()/RAND_MAX;
            sqr_yz = y*y + z*z;
            if (sqr_yz <= 1){
              proc_sum += sqrt(sqr_yz);
            }
        }
        MPI_Allreduce(&proc_sum, &acc_sum, 1, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);
        result = 8.0*acc_sum/(iter*size*block_size);
        if (std::fabs(TARGET - result) < eps){
            stop = 0;
        }
        ++iter;
    }
    double end_time = MPI_Wtime();
    double result_time, time = end_time - start_time;
    MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "True: " << TARGET << std::endl;
        std::cout << "Integral: " << result << std::endl;
        std::cout << "Eps: " << std::abs(TARGET - result) << std::endl;
        std::cout << "N points: " << size * block_size * iter << std::endl;
        std::cout << "Time: " << result_time << std::endl;
    }
    MPI_Finalize();
    return 0;
}
