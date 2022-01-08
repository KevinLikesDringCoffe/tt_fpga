#define N 16
#define DIM 4
#include "dup_stream.h"

typedef float data_t;
typedef struct { data_t data[N]; } pkt;
typedef struct { int indices[DIM]; data_t data; } sp_data;
typedef struct { data_t data[N][N]; } gradient;

template <int DUMMY = 0>
void gemv_stream(
    hls::stream<pkt> &vec_in,
    hls::stream<pkt> &mat_in,
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
    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        pkt temp = mat_in.read();
        for(int j = 0; j < N; j++)
        {
            mat_buffer[i][j] = temp.data[j];
        }
    }

read_vec_stream:
    pkt temp = vec_in.read();
    for(int i = 0; i < N; i++)
    {
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

template <int DUMMY = 0>
void gevm_stream(
    hls::stream<pkt> &vec_in,
    hls::stream<pkt> &mat_in,
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
    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        pkt temp = mat_in.read();
        for(int j = 0; j < N; j++)
        {
            mat_buffer[i][j] = temp.data[j];
        }
    }

read_vec_stream:
    pkt temp = vec_in.read();
    for(int i = 0; i < N; i++)
    {
        vec_buffer[i] = temp.data[i];
    }

gevm_stage:
    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        data_t acc = 0;
        for(int j = 0; j < N; j++)
        {
            data_t temp_sum = mat_buffer[j][i] * vec_buffer[j];
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


template <int DUMMY = 0>
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

) 
{
#pragma HLS dataflow
    hls::stream<pkt> vec_in_pipevm;
    hls::stream<pkt> vec_in_pipemv;
#pragma HLS stream variable=vec_in_pipevm type=fifo depth=16 
#pragma HLS stream variable=vec_in_pipemv type=fifo depth=16 

    dup_stream<pkt>(core1_pipevm, vec_in_pipevm, g2l, 1);
    dup_stream<pkt>(core4_pipemv, vec_in_pipemv, g3r, 1);
    
    
    hls::stream<pkt> vec_2_3_pipevm;
    hls::stream<pkt> vec_3_2_pipemv;
#pragma HLS stream variable=vec_2_3_pipevm type=fifo depth=16 
#pragma HLS stream variable=vec_3_2_pipemv type=fifo depth=16 
    
    gevm_stream(vec_in_pipevm, core2_pipevm, vec_2_3_pipevm);
    gemv_stream(vec_in_pipemv, core3_pipemv, vec_3_2_pipemv);


    hls::stream<pkt> vec_2_3_pipevm_;
    hls::stream<pkt> vec_3_2_pipemv_;
#pragma HLS stream variable=vec_2_3_pipevm_ type=fifo depth=16 
#pragma HLS stream variable=vec_3_2_pipemv_ type=fifo depth=16

    dup_stream<pkt>(vec_2_3_pipevm, vec_2_3_pipevm_, g3l, 1);
    dup_stream<pkt>(vec_3_2_pipemv, vec_3_2_pipemv_, g2r, 1);
    
    
    hls::stream<pkt> vec_3_4_pipevm;
    hls::stream<pkt> vec_2_1_pipemv;
#pragma HLS stream variable=vec_3_4_pipevm type=fifo depth=16 
#pragma HLS stream variable=vec_2_1_pipemv type=fifo depth=16

    gevm_stream(vec_2_3_pipevm_, core3_pipevm, vec_3_4_pipevm);
    gemv_stream(vec_3_2_pipemv_, core2_pipemv, vec_2_1_pipemv);
    
    
    hls::stream<pkt> vec_3_4_pipevm_;
    hls::stream<pkt> vec_2_1_pipemv_;
#pragma HLS stream variable=vec_3_4_pipevm_ type=fifo depth=16 
#pragma HLS stream variable=vec_2_1_pipemv_ type=fifo depth=16

    dup_stream<pkt>(vec_3_4_pipevm, vec_3_4_pipevm_, g4l, 1);
    dup_stream<pkt>(vec_2_1_pipemv, vec_2_1_pipemv_, g1r, 1);
    
    
    pkt temp_vm = vec_3_4_pipevm_.read();
    pkt temp_mv = vec_2_1_pipemv_.read();

}

template <int DUMMY = 0>
void outer(
    data_t scaling,
    hls::stream<pkt> &vec1,
    hls::stream<pkt> &vec2, 
    hls::stream<pkt> &out
)
{  
    data_t vec1_buffer[N];
#pragma HLS array_partition variable = vec1_buffer type = complete dim = 1
    data_t vec2_buffer[N];
#pragma HLS array_partition variable = vec2_buffer type = complete dim = 1
    data_t out_buffer[N][N];
#pragma HLS array_partition variable = out_buffer type = complete dim = 0

#pragma HLS dataflow

read_vec1:
    pkt temp_vec1 = vec1.read();
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        vec1_buffer[i] = temp_vec1.data[i];
    }

read_vec2:
    pkt temp_vec2 = vec2.read();
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        vec2_buffer[i] = temp_vec2.data[i];
    }

outer:
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            out_buffer[i][j] = vec1_buffer[i] * vec2_buffer[j] * scaling;
        }
    }

out_stream:
    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        pkt temp;
        for(int j = 0; j < N; j++)
        {
            temp.data[j] = out_buffer[i][j];
        }
        out.write(temp);
    }
}

template <int DUMMY = 0>
void dot(
    hls::stream<pkt> &vec1,
    hls::stream<pkt> &vec2,
    hls::stream<data_t> &out
)
{
    data_t vec1_buffer[N];
#pragma HLS array_partition variable = vec1_buffer type = complete dim = 1
    data_t vec2_buffer[N];
#pragma HLS array_partition variable = vec2_buffer type = complete dim = 1
    
    pkt temp;
    data_t acc = 0;
#pragma HLS dataflow

read_vec1:
    temp = vec1.read();
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        vec1_buffer[i] = temp.data[i];
    }

read_vec2:
    temp = vec2.read();
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        vec2_buffer[i] = temp.data[i];
    }

mul:
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        data_t t = vec1_buffer[i] * vec2_buffer[i];
        acc += t;
    }

out:
    out.write(acc);
}

template <int DUMMY = 0>
void read_engine(
    hls::stream<int> &slice,
    pkt *core,
    hls::stream<pkt> &core_stream,
    int len
)
{
#pragma HLS dataflow
    int offset = slice.read();
    
    for(int i = 0; i < len; i++)
    {
#pragma HLS loop_tripcount min = 16 max = 16
#pragma HLS pipeline
        pkt temp = core[offset + i];
        core_stream.write(temp);
    }
}

template <int DUMMY = 0>
void write_engine(
    hls::stream<int> &slice,
    pkt *core,
    hls::stream<pkt> &core_stream,
    int len
)
{
#pragma HLS dataflow
    int offset = slice.read();
    for(int i = 0; i < len; i++)
    {
#pragma HLS loop_tripcount min = 16 max = 16
#pragma HLS pipeline
        pkt temp = core_stream.read();
        core[offset + i] = temp;
    }
}

template <int DUMMY = 0>
void coo_sparser(
    sp_data *data_in,
    hls::stream<int> &indice_0,
    hls::stream<int> &indice_1,
    hls::stream<int> &indice_2,
    hls::stream<int> &indice_3,
    hls::stream<data_t> &data,
    int offset
)
{
#pragma HLS data_pack variable = data_in
#pragma HLS dataflow
    sp_data temp = data_in[offset];
    int indice0 = temp.indices[0];
    int indice1 = temp.indices[1];
    int indice2 = temp.indices[2];
    int indice3 = temp.indices[3];

    indice_0.write(indice0);
    indice_1.write(indice1);
    indice_2.write(indice2);
    indice_3.write(indice3);

    data.write(temp.data);
}


template <int DUMMY = 0>
void scale(
    data_t scaling,
    hls::stream<pkt> &vec_in,
    hls::stream<pkt> &vec_out
)
{
#pragma HLS dataflow

    pkt temp = vec_in.read();
    pkt out;

    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        out.data[i] = temp.data[i] * scaling;
    }

    vec_out.write(out);
}


template <int DUMMY = 0>
void update(
    hls::stream<pkt> &core,
    hls::stream<pkt> &grad,
    hls::stream<pkt> &out,
    int len
)
{   
    for(int i = 0; i < len; i++)
    {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 1 max = 16
        pkt core_ = core.read();
        pkt grad_ = grad.read();
        pkt temp;
        for(int j = 0; j < N; j++)
        {
            temp.data[i] = core_.data[i] - grad_.data[i];
        }
        out.write(temp);
    }
}
