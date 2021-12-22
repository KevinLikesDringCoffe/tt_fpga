#define MAX_BUF_SIZE 100
#define TENSOR_BUF 27000

typedef struct{
    int dim;
    int **indices;
    float *t;
    int nnz;

}sp_tensor;

typedef float data_t;

typedef struct { int indices[4]; data_t data; } sp_data;

float dot(float *vec1, float *vec2, int len );

void g_nl(float **core, int *tt_rank, int k, int dim, float *out);

void g_nr(float **core, int *tt_rank, int k, int dim, float *out);

void gemv(float *matrix, float * vector, int m, int n, float *out);

void get_idx(int idx1d, int *size, int dim, int *idx_list);

void gevm(float * matrix, float *vectorT, int m, int n, float *out);

void outer(float *vector, float *vectorT, int m, int n, float *out);

void scale(float *mat, float scalor, int len);

float sumup(float *mat1, float *mat2, int len);

void update_slice(float *slice, int m, int n, float lr, float *grad);

void core2tensor(float **tt_core, int *tt_rank, int *tensor_size, int dim, float *out);

void rand_tensor(int *size, int mode, float *out);

void rand_core(int rn_prev, int rn, int in, float *out);

void ones_tensor(int *size, int mode, float *out);

void rand_sample(float *tensor, int *size ,float mr, sp_tensor *sp_t);

void sgd_engine(sp_data *sp, int nnz, int mode, int *tt_rank, int *tensor_size, float **tt_core, float *out, float lr, int maxiter);

int rand_sample_sp_data(float *tensor, int dim, int *size, float mr, sp_data *sp_t);