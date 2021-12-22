//m*n matrix times n*1 vector, output is m*1 vector
void gemv(float *matrix, float * vector, int m, int n, float *out)
{
    //initialization
    
    for(int i = 0; i < m; i++)
    {
        out[i] = 0;
    }
    

    float acc;

    for(int i = 0; i < m; i++)
    {
        acc = 0;
        for(int j = 0; j < n; j++)
        {
            acc += matrix[i*n + j] * vector[j];
        }
        out[i] = acc;
    }

    return;
}