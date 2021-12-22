// outer product of vector m*1 and vectorT 1*n, output is m*n matrix
void outer(float *vector, float *vectorT, int m, int n, float *out)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            out[i*n + j] = vector[i] * vectorT[j];
        }
    }

    return;
}