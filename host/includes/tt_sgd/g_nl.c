#include "tt_sgd.h"
#include <stdlib.h>

//v-m multiplication chain
void g_nl(float **core, int *tt_rank, int k, int dim, float *out)
{
    float *matrix;
    float *vector_buf = (float *) malloc(sizeof(float) * MAX_BUF_SIZE);

    for(int j = 0; j < tt_rank[1]; j++)
    {
       vector_buf[j] = core[0][j];
       out[j] = core[0][j];
    }

    //float temp_buf[MAX_BUF_SIZE];
    
    for(int i = 1; i < k; i++)
    {
        matrix = core[i];
        gevm(matrix, vector_buf, tt_rank[i], tt_rank[i+1], out);

        //copy
        for(int j = 0; j < tt_rank[i+1]; j++)
        {
            vector_buf[j] = out[j];
        }

        //printf("%f %f %f %f\n", vector_buf[0], vector_buf[1], vector_buf[2], vector_buf[3]);
    }

    //out = temp_buf;
    free(vector_buf);
    return;
}