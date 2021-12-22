//gradient descent step. slice and grad is m*n matrix. done by slice = slice - lr * grad.
void update_slice(float *slice, int m, int n, float lr, float *grad)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            slice[i * n + j] -= lr * grad[i * n + j];
        }
    }

    return;
}