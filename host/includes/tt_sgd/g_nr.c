#include "tt_sgd.h"
#include <stdlib.h>

//m-v multiplication chain
void g_nr(float **core, int *tt_rank, int k, int dim, float *out)
{
    float *matrix;
    float *vector_buf = (float *) malloc(sizeof(float) * MAX_BUF_SIZE);

    for(int j = 0; j < tt_rank[dim - 1]; j++)
    {
       vector_buf[j] = core[dim - 1][j];
       out[j] = core[dim - 1][j];
    }
     
    //float temp_buf[MAX_BUF_SIZE];
    
    for(int i = dim - 2; i > k; i--)
    {
        matrix = core[i];
        gemv(matrix, vector_buf, tt_rank[i], tt_rank[i+1], out);

        //copy
        for(int j = 0; j < tt_rank[i+1]; j++)
        {
            vector_buf[j] = out[j];
            //printf("%f ", vector_buf[j]);
        }

        //printf("\n");
    }

    //out = temp_buf;

    free(vector_buf);
    
    return;
}