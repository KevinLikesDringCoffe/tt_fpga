#include "stdlib.h"
#include "stdio.h"
#include "time.h"

void rand_tensor(int *size, int mode, float *out)
{
    int len = 1;
    for(int i = 0; i < mode; i++)
    {
        len *= size[i];
    }

    srand(time(0));
    for(int i = 0; i < len; i++)
    {
        out[i] = (float) (rand()) / (float) RAND_MAX;
        //printf("%f ", out[i]);
    }

    //debug information
    printf("Tensor with size: ");
    for(int i = 0; i < mode-1; i++)
    {
        printf("%d * ",size[i]);
    }

    printf("%d.\n", size[mode-1]);
    return;
}