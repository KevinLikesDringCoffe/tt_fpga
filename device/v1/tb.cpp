#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <hls_stream.h>

#define N 16
#define M 4
#define MAX_BUF_SIZE 100

typedef float data_t;
typedef struct { data_t data[N]; } pkt;
typedef struct { data_t data[N][N]; } gradient;
typedef struct { int indices[4]; data_t data; } sp_data;

void mvchain_pipe(
    hls::stream<pkt> &core1_pipevm,
    hls::stream<pkt> &core2_pipevm,
    hls::stream<pkt> &core3_pipevm,
    //hls::stream<pkt> &core4_pipevm,

    //hls::stream<pkt> &core1_pipemv,
    hls::stream<pkt> &core2_pipemv,
    hls::stream<pkt> &core3_pipemv,
    hls::stream<pkt> &core4_pipemv,

    hls::stream<pkt> &g2l,
    hls::stream<pkt> &g3l,
    hls::stream<pkt> &g4l,
    
    hls::stream<pkt> &g1r,
    hls::stream<pkt> &g2r,
    hls::stream<pkt> &g3r

);

void outer(
    hls::stream<pkt> &vec1,
    hls::stream<pkt> &vec2, 
    hls::stream<gradient> &out
);

void pipe(
    sp_data *data_in,
    pkt *core1,
    pkt *core2,
    pkt *core3,
    pkt *core4,
    int slc
);

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

float dot(float *vec1, float *vec2, int len )
{
    float acc = 0;
    for(int i = 0; i < len; i++)
    {
        acc += vec1[i] * vec2[i];
    }

    return acc;
}

void g_nr(float **core, int *tt_rank, int k, int dim, float *out)
{
    float *matrix;
    float vector_buf[MAX_BUF_SIZE];

    for(int j = 0; j < tt_rank[dim - 1]; j++)
    {
       vector_buf[j] = core[dim - 1][j];
       out[j] = core[dim - 1][j];
    }
     
    //float temp_buf[MAX_BUF_SIZE];
    
    for(int i = dim - 2; i > k; i--)
    {
        matrix = core[i];
        gemv(matrix, vector_buf, tt_rank[i], tt_rank[i+1], out);

        //copy
        for(int j = 0; j < tt_rank[i+1]; j++)
        {
            vector_buf[j] = out[j];
            //printf("%f ", vector_buf[j]);
        }

        //printf("\n");
    }

    //out = temp_buf;

    return;
}

void g_nl(float **core, int *tt_rank, int k, int dim, float *out)
{
    float *matrix;
    float vector_buf[MAX_BUF_SIZE];

    for(int j = 0; j < tt_rank[1]; j++)
    {
       vector_buf[j] = core[0][j];
       out[j] = core[0][j];
    }

    //float temp_buf[MAX_BUF_SIZE];
    
    for(int i = 1; i < k; i++)
    {
        matrix = core[i];
        gevm(matrix, vector_buf, tt_rank[i], tt_rank[i+1], out);

        //copy
        for(int j = 0; j < tt_rank[i+1]; j++)
        {
            vector_buf[j] = out[j];
        }

        //printf("%f %f %f %f\n", vector_buf[0], vector_buf[1], vector_buf[2], vector_buf[3]);
    }

    //out = temp_buf;

    return;
}

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

float sumup(float *mat1, float *mat2, int len)
{
    float acc = 0;
    for(int i = 0; i < len; i++)
    {
        acc += mat1[i] * mat2[i];
    }

    return acc;
}

void scale(float *mat, float scalor, int len)
{
    for(int i = 0; i < len; i++)
    {
        mat[i] *= scalor;
    }
}

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

void core2tensor(float **tt_core, int *tt_rank, int *tensor_size, int dim, float *out)
{
    int len = 1;
    for(int i = 0; i < dim; i++)
    {
        len *= tensor_size[i];
    }

    int idx_list[dim];
    float *core[dim];
     
    float vec[MAX_BUF_SIZE];

    for(int i = 0; i < len; i++)
    {   
        /*
        int acc1 = 1;
        int acc2 = tensor_size[dim - 1];
        for(int idx = dim - 1; idx > -1; idx--)
        {
            idx_list[idx] = (i % acc2) / acc1;
            acc1 *= tensor_size[idx];
            acc2 *= tensor_size[idx - 1];
        } 
        */

        get_idx(i, tensor_size, dim, idx_list);

        for(int j = dim - 1; j > -1; j--)
        {
            core[j] = tt_core[j] + idx_list[j] * tt_rank[j] * tt_rank[j+1];
        }

        g_nr(core, tt_rank, 0, dim, vec);
        out[i] = dot(vec, core[0], tt_rank[1]);
    }

    return;
}

int main()
{   
    int mode = M;
    int tt_rank[M + 1] = {1, 16, 16, 16, 1};
    int tensor_size[M] = {10, 10, 10, 10};

    float *tt_core[mode];
    float *grad[mode];

    int slc = 10;
    sp_data *sp = (sp_data *) malloc(sizeof(sp_data) * 10);
    for(int i = 0; i < slc; i++)
    {
        sp[i].data = 1;
        for(int j = 0; j < M; j++)
        {
            sp[i].indices[j] = i;
        }
    }

    //allocate space for the tt core and initialization
    for(int i = 0; i < mode; i++)
    {
        tt_core[i] = (float *) malloc(tt_rank[i] * tt_rank[i+1] * tensor_size[i] * sizeof(float));
        rand_core(tt_rank[i], tt_rank[i+1], tensor_size[i], tt_core[i]);
    }
    
    float *out = (float *) malloc(sizeof(float) * 10 * 10 * 10 * 10);
/*
    for(int j = 0 ; j < mode; j ++)
    {
        scale(grad[j], x - y, tt_rank[j] * tt_rank[j+1]);
        update_slice(core[j], tt_rank[j], tt_rank[j+1], 0.01, grad[j]);
    }
*/

    pipe(sp, (pkt *)tt_core[0], (pkt *)tt_core[1], (pkt *)tt_core[2], (pkt *)tt_core[3], slc);
    
    core2tensor(tt_core, tt_rank, tensor_size, M, out);
    /*
    for(int i = 0; i < mode; i++)
    {   
        printf("The %dth core value is:", i);
        for(int j = 0; j < tt_rank[i] * tt_rank[i+1]; j ++)
        {
            printf("%f ", *(tt_core[i] + j));
        }
        printf("\n");
    }*/

    for(int i = 0; i < 10 * 10 * 10 * 10; i++)
    {
        printf(" %f", out[i]);
    }

/* mvchain test code
    //SW part
    hls::stream<pkt> core1_pipevm;
    hls::stream<pkt> core2_pipevm;
    hls::stream<pkt> core3_pipevm;
    //hls::stream<pkt> core4_pipevm;

    //hls::stream<pkt> core1_pipemv;
    hls::stream<pkt> core2_pipemv;
    hls::stream<pkt> core3_pipemv;
    hls::stream<pkt> core4_pipemv;

    hls::stream<pkt> g2l;
    hls::stream<pkt> g3l;
    hls::stream<pkt> g4l;
    
    hls::stream<pkt> g1r;
    hls::stream<pkt> g2r;
    hls::stream<pkt> g3r;

    pkt temp;
    for(int i = 0; i < N; i++)
    {   
        temp.data[i] = core[0][i];
    }
    //core1_pipemv.write(temp);
    core1_pipevm.write(temp);

    for(int i = 0; i < N * N; i += N)
    {
        for(int j = 0; j < N; j++)
        {
            temp.data[j] = core[1][i + j];
        }
        core2_pipemv.write(temp);
        core2_pipevm.write(temp);
    }

    for(int i = 0; i < N * N; i += N)
    {
        for(int j = 0; j < N; j++)
        {
            temp.data[j] = core[2][i + j];
        }
        core3_pipemv.write(temp);
        core3_pipevm.write(temp);
    }

    for(int i = 0; i < N; i++)
    {   
        temp.data[i] = core[3][i];
    }
    core4_pipemv.write(temp);
    //core4_pipevm.write(temp);

    mvchain_pipe(
        core1_pipevm,
        core2_pipevm,
        core3_pipevm,
        //core4_pipevm,

        //core1_pipemv,
        core2_pipemv,
        core3_pipemv,
        core4_pipemv,

        g2l,
        g3l,
        g4l,
        
        g1r,
        g2r,
        g3r
    );

    //hls::stream<grad> grad0;
    //hls::stream<grad> grad1;
    //hls::stream<grad> grad2;
    //hls::stream<grad> grad3;

    pkt grad0 = g1r.read();
    printf("Mode 1, the grad is:\n");
    for(int i = 0; i < N; i++)
    {
        printf("%f ", grad0.data[i]);
    }
    printf("\n");

    //simple comsumer for test
    /*
    pkt grad1_l = g2l.read();
    pkt grad1_r = g2r.read();
    pkt grad2_l = g3l.read();
    pkt grad2_r = g3r.read();
    */
/*
    hls::stream<gradient> grad1;
    hls::stream<gradient> grad2;

    outer(g2l, g2r, grad1);
    outer(g3l, g3r, grad2);

    gradient grad1_ = grad1.read();
    gradient grad2_ = grad2.read();

    printf("Mode 2, the grad is:\n");
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%f ", grad1_.data[i][j]);
        }
    }
    printf("\n");

    printf("Mode 3, the grad is:\n");
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%f ", grad2_.data[i][j]);
        }
    }
    printf("\n");
    
    pkt grad3 = g4l.read();
    printf("Mode 4, the grad is:\n");
    for(int i = 0; i < N; i++)
    {
        printf("%f ", grad3.data[i]);
    }
*/
}