//Tensor is always stored in 1-D form. This function coverts 1d index to normal tensor indices
void get_idx(int idx1d, int *size, int dim, int *idx_list)
{
    int acc1 = 1;
    int acc2 = size[dim - 1];
    for(int idx = dim - 1; idx > -1; idx--)
    {
        idx_list[idx] = (idx1d % acc2) / acc1;
        acc1 *= size[idx];
        acc2 *= size[idx - 1];
    } 

    return;
}