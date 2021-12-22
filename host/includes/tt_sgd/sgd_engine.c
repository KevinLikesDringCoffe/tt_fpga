#include "tt_sgd.h"
#include "time.h"
#include "stdlib.h"
#include "stdio.h"

void sgd_engine(sp_data *sp, int nnz, int mode, int *tt_rank, int *tensor_size, float **tt_core, float *out, float lr, int maxiter)
{

    float *grad[mode];
    clock_t start, stop;

    //allocate space for gradient matrix 
    for(int i = 0; i < mode; i++)
    {
        grad[i] = (float *) malloc(tt_rank[i] * tt_rank[i+1] * sizeof(float));
    }


    float *vec = (float *) malloc(sizeof(float) * MAX_BUF_SIZE);
    float *vecT = (float *) malloc(sizeof(float) * MAX_BUF_SIZE);

    start = clock();

    for(int i = 0; i < maxiter; i++)
    {
        float loss = 0;
        for(int sample = 0; sample < nnz; sample++)
        {
            
            float *core[mode];

            for(int j = mode-1; j > -1; j--)
            {
                core[j] = tt_core[j] + sp[sample].indices[j] * tt_rank[j] * tt_rank[j+1];
            }

            
            for(int j = 0; j < mode; j++)
            {
                if(j == 0)
                {
                    g_nr(core, tt_rank, 0, mode, grad[0]);
                }
                else if(j == mode-1)
                {
                    g_nl(core, tt_rank, mode-1, mode, grad[mode-1]);
                }
                else
                {
                    g_nl(core, tt_rank, j, mode, vecT);
                    g_nr(core, tt_rank, j, mode, vec);
                    outer(vecT, vec, tt_rank[j], tt_rank[j+1], grad[j]);
                }
            }

            float x = sumup(grad[0], core[0], tt_rank[0] * tt_rank[1]);
            float y = sp[sample].data;    
            loss += (x - y) * (x - y);

            for(int j = 0; j < mode; j++)
            {
                scale(grad[j], x - y, tt_rank[j] * tt_rank[j+1]);
                update_slice(core[j], tt_rank[j], tt_rank[j+1], lr, grad[j]);
            }            
        }

        //if(i % 100 == 99)
        {
            printf("Total loss @ iteration %d : %f \n", i+1, loss);
        }
    }

    //recover tensor
    core2tensor(tt_core, tt_rank, tensor_size, mode, out);

    stop = clock();

    float duration = (float) (stop - start) / CLOCKS_PER_SEC;
    
    printf("\ntime cost: %f s\n", duration);

    free(vec);
    free(vecT);
    
    return;
}