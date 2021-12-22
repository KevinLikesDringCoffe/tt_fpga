#include "tt_sgd.h"

//from tt cores to tensor
void core2tensor(float **tt_core, int *tt_rank, int *tensor_size, int dim, float *out)
{
    int len = 1;
    for(int i = 0; i < dim; i++)
    {
        len *= tensor_size[i];
    }

    int idx_list[dim];
    float *core[dim];
     
    float vec[MAX_BUF_SIZE];

    for(int i = 0; i < len; i++)
    {   
        /*
        int acc1 = 1;
        int acc2 = tensor_size[dim - 1];
        for(int idx = dim - 1; idx > -1; idx--)
        {
            idx_list[idx] = (i % acc2) / acc1;
            acc1 *= tensor_size[idx];
            acc2 *= tensor_size[idx - 1];
        } 
        */

        get_idx(i, tensor_size, dim, idx_list);

        for(int j = dim - 1; j > -1; j--)
        {
            core[j] = tt_core[j] + idx_list[j] * tt_rank[j] * tt_rank[j+1];
        }

        g_nr(core, tt_rank, 0, dim, vec);
        out[i] = dot(vec, core[0], tt_rank[1]);
    }

    return;
}