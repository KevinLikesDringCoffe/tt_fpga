#include "tt_sgd.h"
#include "stdlib.h"
#include "stdio.h"

//this function samples the original tensor and returns random indices
void rand_sample(float *tensor, int *size ,float mr, sp_tensor *sp_t)
{
    int dim = sp_t->dim;
    int len = 1;
    for(int i = 0; i < dim; i++)
    {
        len *= size[i];
    }

    float *t = (float *) malloc(len * sizeof(float));
    rand_tensor(size, dim, t);
    
    int nnz = 0;

    for(int  i = 0; i < len; i++){
        if(t[i] > mr)
        {   
            int acc1 = 1;
            int acc2 = size[dim - 1]; 
            for(int idx = dim - 1; idx > -1; idx--)
            {
                sp_t->indices[idx][nnz] = (i % acc2) / acc1;
                acc1 *= size[idx];
                acc2 *= size[idx - 1];
                //printf("%d ", sp_t->indices[idx][nnz]);
            }
            
            sp_t->t[nnz] = tensor[i];
            //printf("%d, %f\n" ,i, sp_t->t[nnz]);
            nnz++;
            //t[i] = 1;
        }
        else
        {
            //t[i] = 0;
        }
    }

    sp_t->nnz = nnz; 
    //free(t);
    return;
}