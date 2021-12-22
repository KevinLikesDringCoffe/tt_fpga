//summation for element-wise prod
float sumup(float *mat1, float *mat2, int len)
{
    float acc = 0;
    for(int i = 0; i < len; i++)
    {
        acc += mat1[i] * mat2[i];
    }

    return acc;
}