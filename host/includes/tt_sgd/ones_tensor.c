#include "stdlib.h"
#include "stdio.h"

void ones_tensor(int *size, int mode, float *out)
{
    int len = 1;
    for(int i = 0; i < mode; i++)
    {
        len *= size[i];
    }

    for(int i = 0; i < len; i++)
    {
        out[i] = (i % 5);
        //printf("%f ", out[i]);
    }

    // information
    printf("Tensor with size: ");
    for(int i = 0; i < mode; i++)
    {
        printf("%d * ",size[i]);
    }
    
    return;
}