//1*m vector times m*n matrix, output is 1*n vector
void gevm(float * matrix, float *vectorT, int m, int n, float *out)
{
    //initialization
    for(int i = 0; i < n; i++)
    {
        out[i] = 0;
    }

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            out[j] += vectorT[i] * matrix[i*n + j];
        }
    }

    return;
}