#include <mpi.h>
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
    int strip_size, size;
    int rank, rc;
    FILE * fp;
    int *data = NULL, **vectors=NULL, *stripdata=NULL, **strip_vectors = NULL;
    double *res = NULL, *res_strip = NULL;
    long int diff;
    struct timespec mt1, mt2;
    MPI_Datatype strip;
    MPI_Status *status;

    // INIT
    if ((rc = MPI_Init(&argc, &argv)) != MPI_SUCCESS)
    {

        fprintf(stderr, "Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    //printf("rnk=%d\tstarted\n", rank);

    // Read data for rank==0
    if (rank == 0){
        if (argc < 5){
            printf("Usage: %s M N input_file output_file\n", argv[0]);
            exit(1);
        }
        if(sscanf(argv[1], "%d", &M) != 1 || sscanf(argv[2], "%d", &N) != 1 || M < 0 || N < 0) {
            printf("M and N operation requires two integer parameters >= 0.\n");
            exit(1);
        }
        
        strip_size = M / size;
        //printf("strip_size=%d\n", strip_size);
        
        //printf("M=%d\n", M);
        //printf("N=%d\n", N);

        fp = fopen(argv[3], "r");
    
        if(!fp){ 
            puts("Source file not found");
            exit(1);
        }

        data = (int*) malloc(M * N * sizeof(int));
        vectors = (int**) malloc(M * sizeof(int*));
        for (i = 0; i < M; i++) {
            vectors[i] = &(data[i*N]);
            for(j=0; j < N && !feof(fp); j++){
                fscanf(fp, "%d",&vectors[i][j]);
            }
            if (feof(fp) && j<N-1){
                puts("Source file too small.");
                exit(1);
            }
        }
        fclose(fp);
        //puts("Read source file finished.");

        res = (double*) malloc(M * sizeof(double));
        
        fp = fopen(argv[4], "w");
        if(!fp){ 
            puts("Destination file cannot be opened for writing.");
            exit(1);
        } else {
            //puts("Opened dest file finished.");
        }

        // Start timer only for rank==0
        clock_gettime(CLOCK_MONOTONIC, &mt1);
    }

    //MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&strip_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //sleep(rank*2);
    stripdata = (int *)malloc(sizeof(int)*strip_size*N);
    strip_vectors = (int**) malloc(strip_size * sizeof(int*));
    for(i= 0; i< strip_size; i++) {
        strip_vectors[i] = &(stripdata[i*N]);;
    }
    
    res_strip = (double*) malloc(strip_size * sizeof(double));

    //printf("rnk=%d\tallocated mem\n", rank);

    /* defining a datatype for sub-matrix */
    MPI_Type_vector(strip_size, N, N, MPI_INT, &strip);
    MPI_Type_commit(&strip);

    
    //SEND
    //printf("rnk=%d\tgoing to scatter\n", rank);
    MPI_Scatter(data, 1, strip, &(strip_vectors[0][0]), 1, strip, 0, MPI_COMM_WORLD);
    //printf("rnk=%d\tscattered\n", rank);
   
    //sleep(2*rank);
    // COUNT
    for(i = 0; i < strip_size; i++){
        res_strip[i] = vector_len(strip_vectors[i], N);
        
    }
    //printf("rnk=%d\tcounted lengths\n", rank);

    // COLLECT RES
    //printf("rnk=%d\tgoing to gather\n", rank);
    MPI_Gather(res_strip, strip_size, MPI_DOUBLE, res, strip_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //printf("rnk=%d\tgathered\n", rank);


    if (rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &mt2);

        diff = 1000000000*(mt2.tv_sec - mt1.tv_sec)+(mt2.tv_nsec - mt1.tv_nsec);
        printf("%ld ns\n", diff);
        printf("%lf s\n", ((double)diff)/1000000000.0);

        for(i = 0; i < M; i++){
            fprintf(fp, "%0.3lf", res[i]);
            if (i<M-1){
                fprintf(fp, "\n");
            }
        }
        fclose(fp); 
        //puts("Dest file ready");
        
        free(data);
        free(res);
    }

    MPI_Type_free(&strip);
    free(strip_vectors);
    free(stripdata);
    
    free(res_strip);
    //printf("rnk=%d\tfinished\n", rank);
    MPI_Finalize();
    return 0;
}

