#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREADS 1024

__device__ double vector_len_cuda(int* x, int N) {
    double sum = 0;
    int i;
    for (i = 0; i < N; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

__global__ void vector_count(int* data, double* res, int M, int N) {
    int vector_number = blockIdx.x * blockDim.x + threadIdx.x;
    if ((vector_number < M))
        res[vector_number] = vector_len_cuda(&(data[vector_number * N]), N);
}

double vector_len(int* x, int N) {
    double sum = 0;
    int i;
    for (i = 0; i < N; i++) {
        sum += pow(x[i], 2);
    }
    return sqrt(sum);
}

int main(int argc, char* argv[]) {
    int M = 5242880, N = 64, i, j;
    int strip_size, size;
    int rank, rc;
    FILE* fp;
    int* data = NULL, ** vectors = NULL, * stripdata = NULL, ** strip_vectors = NULL;
    double* res = NULL, * res_strip = NULL;
    //long int diff;
    // struct timespec mt1, mt2;
    MPI_Datatype strip;
    std::chrono::steady_clock::time_point start;

    /*int threads = THREADS;*/

    // INIT
    if ((rc = MPI_Init(&argc, &argv)) != MPI_SUCCESS)
    {

        fprintf(stderr, "Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {

        if (argc < 5) {
            printf("Usage: %s M N input_file output_file\n", argv[0]);
            exit(1);
        }
        if (sscanf(argv[1], "%d", &M) != 1 || sscanf(argv[2], "%d", &N) != 1 || M < 0 || N < 0) {
            printf("M and N operation requires two integer parameters >= 0.\n");
            exit(1);
        }

        strip_size = M / size;

        fp = fopen(/*"input.txt"*/argv[3], "r");

        if (!fp) {
            puts("Source file not found");
            exit(1);
        }

        // data = (int*) malloc(M * N * sizeof(int));
        data = (int*)malloc(M * N * sizeof(int));
        vectors = (int**)malloc(M * sizeof(int*));

        //puts("before");

        for (i = 0; i < M; i++) {
            //printf("i*N=%d\n", i * N);
            vectors[i] = &(data[i * N]);
            for (j = 0; j < N && !feof(fp); j++) {
                fscanf(fp, "%d", &vectors[i][j]);
            }
            if (feof(fp) && j < N - 1) {
                puts("Source file too small.");
                exit(1);
            }
        }
        fclose(fp);

        //puts("after read");
        // res = (double*) malloc(M * sizeof(double));
        res = (double*)malloc(M * sizeof(double));

        fp = fopen(/*"output.txt"*/argv[4], "w");
        if (!fp) {
            puts("Destination file cannot be opened for writing.");
            exit(1);
        }


        /*if (threads > M)
            threads = M;*/

            //clock_gettime(CLOCK_MONOTONIC, &mt1);
        start = std::chrono::steady_clock::now();
    }
    


    //exchange
    //MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&strip_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cudaMallocManaged(&res_strip, strip_size * sizeof(double), cudaMemAttachGlobal);
        cudaMallocManaged((void**)&stripdata, strip_size * N * sizeof(int), cudaMemAttachGlobal);
    }
    else
    {
        res_strip = (double*)malloc(strip_size * sizeof(double));
        stripdata = (int*)malloc(sizeof(int) * strip_size * N);
    }
    strip_vectors = (int**)malloc(strip_size * sizeof(int*));
    for (i = 0; i < strip_size; i++) {
        strip_vectors[i] = &(stripdata[i * N]);
    }

    //printf("rnk=%d\tallocated mem\n", rank);

    /* defining a datatype for sub-matrix */
    MPI_Type_vector(strip_size, N, N, MPI_INT, &strip);
    MPI_Type_commit(&strip);


    //SEND
    //printf("rnk=%d\tgoing to scatter\n", rank);
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(data, 1, strip, &(strip_vectors[0][0]), 1, strip, 0, MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);
    //CALC
    if (rank == 0){
        dim3 grid(strip_size, 1, 1);
        dim3 block(N, 1, 1);
        vector_count << <grid, block >> > (stripdata, res_strip, strip_size, N);
        cudaDeviceSynchronize();
    }
    else {
        for (i = 0; i < strip_size; i++) {
            res_strip[i] = vector_len(strip_vectors[i], N);

        }
    }
    
    //COLLECT
    MPI_Gather(res_strip, strip_size, MPI_DOUBLE, res, strip_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        //clock_gettime(CLOCK_MONOTONIC, &mt2);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //puts("after count");
        //diff = 1000000000 * (mt2.tv_sec - mt1.tv_sec) + (mt2.tv_nsec - mt1.tv_nsec);


        std::chrono::steady_clock::duration duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        //printf("%ld ns\n", diff);
        long double calcs_time_s = duration.count() * 1e-9;
        std::cout << calcs_time_s << " s" << std::endl;

        /*for(i = 0; i < M; i++){
            fprintf(fp, "%0.3lf", res[i]);
            if (i<M-1){
                fprintf(fp, "\n");
            }
        }*/

        free(vectors);
        // free(data);
        // (res);

        free(data);
        free(res);
    }
    
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&strip);
    free(strip_vectors);
    if (rank == 0) {
        cudaFree(stripdata);
        cudaFree(res_strip);
    }
    else {
        free(stripdata);

        free(res_strip);
    }
    
    //printf("rnk=%d\tfinished\n", rank);
    MPI_Finalize();

    return 0;
}

