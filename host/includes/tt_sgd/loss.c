#include "tt_sgd.h"

float loss(float **tt_core, int *tt_rank, int *tensor_size, int dim, sp_data *sp, int nnz)
{
    float acc = 0;
    
    for(int i = 0; i < nnz; i++)
    {
        int indices[dim];
        float *core[dim];
        float buf[MAX_BUF_SIZE];
        //get_indices
        for(int j = 0; j < dim; j++)
        {
            indices[j] = sp[i].indices[j];
        }

        float y = sp[i].data;
        for(int j = 0; j < dim; j++)
        {
            core[j] = tt_core[j] + indices[j] * tt_rank[j] * tt_rank[j+1];
        }

        g_nr(core, tt_rank, 0, dim, buf);
        float x = dot(core[0], buf, tt_rank[1]);
        acc += (x - y) * (x - y);
    }

    return acc;
}