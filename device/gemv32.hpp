#define N 32
#define BLOCKS 2
#define DWIDTH N / BLOCKS

typedef float data_t;

typedef struct{ data_t data[DWIDTH]; } pkt;


void gemv32(
    data_t mat[N][N], 
    data_t vec[N], 
    data_t out[N]
)
{
//#pragma HLS interface mode=ap_ctrl_chain port=return 
#pragma HLS array_partition variable=mat complete dim=2
//#pragma HLS array_partition variable=mat type=cyclic factor=2 dim=1
#pragma HLS array_partition variable=vec complete dim=1
#pragma HLS array_partition variable=out complete dim=1

row:
    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
//#pragma HLS unroll factor=2
        data_t temp = 0;

        /*
        data_t temp[N];
#pragma HLS array_partition variable=temp complte dim=1
    reset:
        for(int j = 0; j < N; j++)
        {
            temp[j] = 0;
        }
*/
    col:
        for(int j = 0; j < N; j++)
        {
            //temp[j] += mat[i][j] * vec[j];
            //std::cout << mat[i][j] << " ";
            temp += mat[i][j] * vec[j];
        }

        out[i] = temp;
        //std::cout << temp << " ";
/*
    sumup:
        for(int j = 0; j < N; j++)
        {
            out[i] += temp[j];
        }
*/
    }
}

void gemv_stream(
    hls::stream<pkt> &stream_mat,
    hls::stream<pkt> &stream_vec,
    hls::stream<pkt> &stream_out
)
{
    data_t mat[N][N];
    data_t vec[N];
    data_t out[N];
#pragma HLS dataflow
load_mat:
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j+=DWIDTH)
        {
            pkt temp = stream_mat.read();
            for(int idx = 0; idx < DWIDTH; idx++)
            {
                mat[i][j + idx] = temp.data[idx];
            }
        }
    }

load_vec:
    for(int i = 0; i < N; i += DWIDTH)
    {
        pkt temp = stream_vec.read();
        for(int idx = 0; idx < DWIDTH; idx++)
        {
            vec[i + idx] = temp.data[idx];
        }
    }

gemv32:
    gemv32(mat, vec, out);

write_out_stream:
    for(int i = 0; i < N; i+= DWIDTH)
    {
        pkt temp;
        for(int idx = 0; idx < DWIDTH; idx++)
        {
            temp.data[idx] = out[i + idx];
        }
        stream_out.write(temp);
    }
}
