#define N 32
#define BLOCKS 2
#define DWIDTH N / BLOCKS

typedef float data_t;

typedef struct{ data_t data[DWIDTH]; } pkt;

void gemv_stream(
    hls::stream<pkt> &stream_mat,
    hls::stream<pkt> &stream_vec,
    hls::stream<pkt> &stream_out
);

void gemv_pipeline(
    pkt *vec,
    pkt *mat1,
    pkt *mat2,
    pkt *mat3,
    pkt *mat4,
    pkt *out,
    int slc
);

void gemv_stream(
    hls::stream<pkt> &stream_mat,
    hls::stream<pkt> &stream_vec,
    hls::stream<pkt> &stream_out
)
{
    float vec_buf[N];
#pragma HLS array_partition variable = vec_buf type = complete dim = 1
    float out_buf[N];
#pragma HLS array_partition variable = out_buf type = complete dim = 1

#pragma HLS dataflow
read_vec:
    for(int i = 0; i < N; i += DWIDTH)
    {
#pragma HLS pipeline
        pkt temp = stream_vec.read();
        for(int j = 0; j < DWIDTH; j++)
        {
            vec_buf[i + j] = temp.data[j];
        }
    }

    float partial_sum[N * BLOCKS];
#pragma HLS array_partition variable = partial_sum dim = 1

streaming_mul:
    for(int i = 0, blc = 0; i < BLOCKS * N; i++, blc+=DWIDTH)
    {
#pragma HLS pipeline II = 1 rewind
        if(blc == N)
        {
            blc = 0;
        }

        pkt temp = stream_mat.read();
        float acc = 0;
        for(int j = 0; j < DWIDTH; j++)
        {
            acc += temp.data[j] * vec_buf[blc + j];
            //std::cout << vec_buf[blc + j] << " ";
        }

        partial_sum[i] = acc;
    }

summup:
    for(int i = 0, j = 0; i < BLOCKS * N; i+=BLOCKS, j++)
    {
#pragma HLS unroll
        out_buf[j] = partial_sum[i] + partial_sum[i+1];
    }

out:
    for(int i = 0; i < N; i+= DWIDTH)
    {
#pragma HLS pipeline
        pkt temp;
        for(int j = 0; j < DWIDTH; j++)
        {
            temp.data[j] = out_buf[i + j];
            //std::cout << out_buf[i + j] << " ";
        }
        stream_out.write(temp);
        
    }
}


void gevm_stream(
    hls::stream<pkt> &stream_mat,
    hls::stream<pkt> &stream_vec,
    hls::stream<pkt> &stream_out
)
{
    float vec_buf[N];
#pragma HLS array_partition variable = vec_buf type = complete dim = 1
    float inter_buf[N][N];
#pragma HLS array_partition variable = inter_buf type = complete dim = 0
    float out_buf[N];
#pragma HLS array_partition variable = out_buf type = complete dim = 1

#pragma HLS dataflow
read_vec:
    for(int i = 0; i < N; i += DWIDTH)
    {
#pragma HLS pipeline
        pkt temp = stream_vec.read();
        for(int j = 0; j < DWIDTH; j++)
        {
            vec_buf[i + j] = temp.data[j];
        }
    }

stream_mul:
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < N; j += DWIDTH)
        {
#pragma HLS pipeline
            pkt temp = stream_mat.read();
            for(int k = 0; k < DWIDTH; k++)
            {
                inter_buf[i][j + k] = temp.data[k] * vec_buf[i];
            }
        }
    }

/*
    for(int i = 0, blc = 0; i < BLOCKS * N; i++, blc+=DWIDTH)
    {
#pragma HLS pipeline II = 1 
        if(blc == N)
        {
            blc = 0;
        }

        pkt temp = stream_mat.read();

        for(int j = 0; j < DWIDTH; j++)
        {
            out_buf[blc + j] += temp.data[j] * vec_buf[blc + j];
            //std::cout << vec_buf[blc + j] << " ";
        }
    }
*/

summup:
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        float temp = 0;
        for(int j = 0; j < N; j++)
        {
            temp += inter_buf[i][j];
        }
        out_buf[i] = temp;
    }

out:
    for(int i = 0; i < N; i+= DWIDTH)
    {
#pragma HLS pipeline
        pkt temp;
        for(int j = 0; j < DWIDTH; j++)
        {
            temp.data[j] = out_buf[i + j];
            //std::cout << out_buf[i + j] << " ";
        }
        stream_out.write(temp);
        
    }
}
