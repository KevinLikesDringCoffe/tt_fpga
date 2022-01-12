typedef struct { float data[N]; } pkt;
typedef struct { float data[N][N]; } mat;

void gemv_opt(
    hls::stream<pkt> &vec_in,
    hls::stream<mat> &mat_in,
    hls::stream<pkt> &vec_out
)
{
    data_t vec_buffer[N];
#pragma HLS array_partition variable = vec_buffer type = complete dim = 1
    data_t mat_buffer[N][N];
#pragma HLS array_partition variable = mat_buffer type = complete dim = 0
    data_t out_buffer[N];
#pragma HLS array_partition variable = out_buffer type = complete dim = 1

#pragma HLS dataflow
read_mat_stream:
    mat t = mat_in.read();
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        for(int j = 0; j < N; j++)
        {
#pragma HLS unroll
            mat_buffer[i][j] = t.data[i][j];
        }
    }

read_vec_stream:
    pkt temp = vec_in.read();
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        vec_buffer[i] = temp.data[i];
    }

gemv_stage:
    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        data_t acc = 0;
        for(int j = 0; j < N; j++)
        {
            data_t temp_sum = mat_buffer[i][j] * vec_buffer[j];
            acc = acc + temp_sum;
        }
        out_buffer[i] = acc;
    }

write_out_stream:
    pkt out;
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        out.data[i] = out_buffer[i];
    }
    vec_out.write(out);
}