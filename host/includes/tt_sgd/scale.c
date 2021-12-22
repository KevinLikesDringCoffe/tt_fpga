//scalor-matrix multiplication
void scale(float *mat, float scalor, int len)
{
    for(int i = 0; i < len; i++)
    {
        mat[i] *= scalor;
    }
}