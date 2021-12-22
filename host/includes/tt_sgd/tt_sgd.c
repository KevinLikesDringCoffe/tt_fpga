#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "tt_sgd.h"
#define M 4

int main()
{
    clock_t start, stop;
 
    float mr = 0.8;
    float margin = 0.05;
    int mode = M;
    int tt_rank[M + 1] = {1, 16, 16, 16, 1};
    int tensor_size[M] = {50, 50, 50, 50};

    float *tt_core[mode];
    float *grad[mode];

    //sptensor attribute

    int len = 1;

    for(int i = 0; i < mode; i++)
    {
        len *= tensor_size[i];
    }

    float *t = (float *) malloc(len * sizeof(float));

    ones_tensor(tensor_size, mode, t);

    //allocate space for coo data
    sp_data *sp = (sp_data *) malloc((int) (sizeof(sp_data) * len * (mr + margin)));

    //have some bugs when missing rate is low
    int nnz = rand_sample_sp_data(t, mode, tensor_size, mr, sp);

    //debug info  
#ifdef VERBOSE  
    for(int i = 0; i < nnz; i++)
    {
        printf("(");
        for(int j = 0; j < mode; j++)
        {
            printf("%d,", sp[i].indices[j]);
        }
        printf(")  :  %f\n", sp[i].data);
    }
#endif

    //allocate space for the tt core and initialization
    for(int i = 0; i < mode; i++)
    {
        tt_core[i] = (float *) malloc(tt_rank[i] * tt_rank[i+1] * tensor_size[i] * sizeof(float));
        rand_core(tt_rank[i], tt_rank[i+1], tensor_size[i], tt_core[i]);
    }

    //allocate space for recovered tensor
    float *out = (float *) malloc(len * sizeof(float));

    sgd_engine(sp, nnz, mode, tt_rank, tensor_size, tt_core, out, 0.0001, 1000);

    printf("The output is ");

    for(int i = 0; i < len; i++)
    {
        printf("%f ", out[i]);
    }

    printf("\n");
    
    return 0;
}

