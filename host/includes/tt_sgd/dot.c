//dot product of two vectors
float dot(float *vec1, float *vec2, int len )
{
    float acc = 0;
    for(int i = 0; i < len; i++)
    {
        acc += vec1[i] * vec2[i];
    }

    return acc;
}