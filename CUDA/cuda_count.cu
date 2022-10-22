#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREADS 1024

__device__ double vector_len(int* x, int N) {
    double sum = 0;
    int i;
    for (i = 0; i < N; i++) {
        sum += x[i]*x[i];
    }
    return sqrt(sum);
}

__global__ void vector_count(int* data, double* res, int M, int N) {
    int vector_number = blockIdx.x * blockDim.x + threadIdx.x;
    if ((vector_number < M))
        res[vector_number] = vector_len(&(data[vector_number * N]), N);
}

int main(int argc, char* argv[]) {
    int M= 5242880, N=64, i, j;
    FILE* fp;
    int** vectors = NULL, * data = NULL;
    double* res = NULL;
    //long int diff;
    // struct timespec mt1, mt2;

    /*int threads = THREADS;*/

    if (argc < 5) {
        printf("Usage: %s M N input_file output_file\n", argv[0]);
        exit(1);
    }
    if (sscanf(argv[1], "%d", &M) != 1 || sscanf(argv[2], "%d", &N) != 1 || M < 0 || N < 0) {
        printf("M and N operation requires two integer parameters >= 0.\n");
        exit(1);
    }
    fp = fopen(/*"input.txt"*/argv[3], "r");

    if (!fp) {
        puts("Source file not found");
        exit(1);
    }

    // data = (int*) malloc(M * N * sizeof(int));
    cudaMallocManaged((void**)&data, M * N * sizeof(int), cudaMemAttachGlobal);
    if (data == NULL) {
        puts("mem not allocated");
        exit(0);
    }
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
    cudaMallocManaged(&res, M * sizeof(double), cudaMemAttachGlobal);

    fp = fopen(/*"output.txt"*/argv[4], "w");
    if (!fp) {
        puts("Destination file cannot be opened for writing.");
        exit(1);
    }


    /*if (threads > M)
        threads = M;*/

    //clock_gettime(CLOCK_MONOTONIC, &mt1);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();


    dim3 grid(M, 1, 1);
    dim3 block(N, 1, 1);
    vector_count <<<grid, block >>> (data, res, M, N);
    cudaDeviceSynchronize();

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

    cudaFree(data);
    cudaFree(res);

    return 0;
}

