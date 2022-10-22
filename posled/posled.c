#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double vector_len(int *x, int N){
    double sum=0;
    int i;
    for(i=0;i<N;i++){
        sum += pow(x[i], 2);
    }
    return sqrt(sum);
}

int main(int argc, char* argv[]){
    int M, N, i, j;
    FILE * fp;
    int **vectors = NULL;
    double *res = NULL;
    long int diff;
    struct timespec mt1, mt2;
    if (argc < 5){
        printf("Usage: %s M N input_file output_file\n", argv[0]);
        exit(1);
    }
    if(sscanf(argv[1], "%d", &M) != 1 || sscanf(argv[2], "%d", &N) != 1 || M < 0 || N < 0) {
        printf("M and N operation requires two integer parameters >= 0.\n");
        exit(1);
    }
    fp = fopen(argv[3], "r");
 
    if(!fp){ 
        puts("Source file not found");
        exit(1);
    }

    vectors = (int**) malloc(M * sizeof(int*));
    for (i = 0; i < M; i++) {
        vectors[i] = (int*) malloc(N * sizeof(int));
        for(j=0; j < N && !feof(fp); j++){
            fscanf(fp, "%d",&vectors[i][j]);
        }
        if (feof(fp) && j<N-1){
            puts("Source file too small.");
            exit(1);
        }
    }
    fclose(fp);

    res = (double*) malloc(M * sizeof(double));
    
    fp = fopen(argv[4], "w");
    if(!fp){ 
        puts("Destination file cannot be opened for writing.");
        exit(1);
    }


    clock_gettime(CLOCK_MONOTONIC, &mt1);

    for(i = 0; i < M; i++){
        res[i] = vector_len(vectors[i], N);
        
    }
    
    clock_gettime(CLOCK_MONOTONIC, &mt2);

    diff = 1000000000*(mt2.tv_sec - mt1.tv_sec)+(mt2.tv_nsec - mt1.tv_nsec);

    printf("%ld ns\n", diff);
    printf("%lf s\n", ((double)diff)/1000000000.0);

   /*  for(i = 0; i < M; i++){
        fprintf(fp, "%0.3lf", res[i]);
        if (i<M-1){
            fprintf(fp, "\n");
        }
    } */

    for (i = 0; i < M; i++) {
        free(vectors[i]);
    }
    free(vectors);
    free(res);

    return 0;
}

