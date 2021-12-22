#include "stdlib.h"
#include "stdio.h"
#include "time.h"

void rand_core(int rn_prev, int rn, int in, float *out)
{
    int len;
    len = rn_prev * rn * in;
    srand(time(0));
    for(int i = 0; i < len; i++)
    {
        out[i] = (float) (rand()) / (float)RAND_MAX;
    }

    printf("Tensor core with size: %d * %d * %d .\n", in, rn_prev, rn);

    return;
}