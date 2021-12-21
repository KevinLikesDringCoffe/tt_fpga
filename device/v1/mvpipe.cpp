#include <hls_stream.h>

#define N 16
typedef float data_t;
typedef struct { data_t data[N]; } pkt;
typedef struct { int indices[4]; data_t data; } sp_data;

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

void dup_stream(
    hls::stream<pkt> &in,
    hls::stream<pkt> &dup1,
    hls::stream<pkt> &dup2
)
{
#pragma HLS pipeline
    pkt temp = in.read();
    dup1.write(temp);
    dup2.write(temp);
}

void dup_stream_sp(
    hls::stream<sp_data> &in,
    hls::stream<sp_data> &dup1,
    hls::stream<sp_data> &dup2
)
{
#pragma HLS pipeline
    sp_data temp = in.read();
    dup1.write(temp);
    dup2.write(temp);
}

void dup_stream_int(
    hls::stream<int> &in,
    hls::stream<int> &dup1,
    hls::stream<int> &dup2
)
{
#pragma HLS pipeline
    int temp = in.read();
    dup1.write(temp);
    dup2.write(temp);
}
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

    for(int i = 0; i < 1; i++)
    {
        dup_stream(core1_pipevm, vec_in_pipevm, g2l);
        dup_stream(core4_pipemv, vec_in_pipemv, g3r);
    }
    
    hls::stream<pkt> vec_2_3_pipevm;
    hls::stream<pkt> vec_3_2_pipemv;

    for(int i = 0; i < 1; i++)
    {
        gevm_stream(vec_in_pipevm, core2_pipevm, vec_2_3_pipevm);
        gemv_stream(vec_in_pipemv, core3_pipemv, vec_3_2_pipemv);
    }

    hls::stream<pkt> vec_2_3_pipevm_;
    hls::stream<pkt> vec_3_2_pipemv_;

    for(int i = 0; i < 1; i++)
    {
        dup_stream(vec_2_3_pipevm, vec_2_3_pipevm_, g3l);
        dup_stream(vec_3_2_pipemv, vec_3_2_pipemv_, g2r);
    }
    
    hls::stream<pkt> vec_3_4_pipevm;
    hls::stream<pkt> vec_2_1_pipemv;

    for(int i = 0; i < 1; i++)
    {
        gevm_stream(vec_2_3_pipevm_, core3_pipevm, vec_3_4_pipevm);
        gemv_stream(vec_3_2_pipemv_, core2_pipemv, vec_2_1_pipemv);
    }
    
    hls::stream<pkt> vec_3_4_pipevm_;
    hls::stream<pkt> vec_2_1_pipemv_;
    
    for(int i = 0; i < 1; i++)
    {
        dup_stream(vec_3_4_pipevm, vec_3_4_pipevm_, g4l);
        dup_stream(vec_2_1_pipemv, vec_2_1_pipemv_, g1r);
    }
    
    for(int i = 0; i < 1; i++)
    {
        pkt temp_vm = vec_3_4_pipevm_.read();
        pkt temp_mv = vec_2_1_pipemv_.read();
    }
}


typedef struct { data_t data[N][N]; } gradient;
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
/*
    gradient temp;
    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        for(int j = 0; j < N; j++)
        {
            temp.data[i][j] = out_buffer[i][j];
        }
    }
    out.write(temp);
*/
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
#pragma HLS pipeline
        data_t t = vec1_buffer[i] * vec2_buffer[i];
        acc += t;
    }

out:
    out.write(acc);
}

void summup(
    hls::stream<pkt> &grad,
    hls::stream<pkt> &core,
    hls::stream<data_t> &out
)
{
    data_t acc = 0;
    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        pkt grad_ = grad.read();
        pkt core_ = core.read();

        for(int j = 0; j < N; j++)
        {
            acc += grad_.data[j] * core_.data[j];
        }
    }

    out.write(acc);
}

void data_fetch_engine(
    //memory interface
    sp_data *data_in,
    pkt *core1,
    pkt *core2,
    pkt *core3,
    pkt *core4,
    
    //stream interface
    hls::stream<pkt> &stream_core1,
    hls::stream<pkt> &stream_core2,
    hls::stream<pkt> &stream_core3,
    hls::stream<pkt> &stream_core4,
/*
    hls::stream<pkt> &core1_update,
    hls::stream<pkt> &core2_update,
    hls::stream<pkt> &core3_update,
    hls::stream<pkt> &core4_update,
    hls::stream<sp_data> &update_indices,
*/
    hls::stream<sp_data> &y
)
{

#pragma HLS interface port = data_in mode = m_axi offset = slave bundle = gmem0
#pragma HLS interface port = core1   mode = m_axi offset = slave bundle = gmem1
#pragma HLS interface port = core2   mode = m_axi offset = slave bundle = gmem2
#pragma HLS interface port = core3   mode = m_axi offset = slave bundle = gmem3
#pragma HLS interface port = core4   mode = m_axi offset = slave bundle = gmem4

#pragma HLS interface s_axilite port = data_in
#pragma HLS interface s_axilite port = core1
#pragma HLS interface s_axilite port = core2
#pragma HLS interface s_axilite port = core3
#pragma HLS interface s_axilite port = core4




    sp_data temp = *data_in;
    y.write(temp);

    int index_0 = temp.indices[0];
    int index_1 = temp.indices[1];
    int index_2 = temp.indices[2];
    int index_3 = temp.indices[3];

    int core1_offset = index_0;
    int core2_offset = index_1 * N;
    int core3_offset = index_2 * N;
    int core4_offset = index_3;

read_core1:
    for(int i = 0; i < 1; i++)
    {
#pragma HLS pipeline
        pkt t = *(core1 + core1_offset + i);
        stream_core1.write(t);
    } 

read_core2:
    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        pkt t = *(core2 + core2_offset + i);
        stream_core2.write(t);
    } 

read_core3:
    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        pkt t = *(core3 + core3_offset + i);
        stream_core3.write(t);
    } 

read_core4:
    for(int i = 0; i < 1; i++)
    {
#pragma HLS pipeline
        pkt t = *(core4 + core4_offset + i);
        stream_core4.write(t);
    } 
}


void read_engine(
    hls::stream<int> &slice,
    pkt *core,
    hls::stream<pkt> &core_stream,
    int len
)
{
    int offset = slice.read();
    
    for(int i = 0; i < len; i++)
    {
#pragma HLS pipeline
        pkt temp = core[offset + i];
        core_stream.write(temp);
    }
}

void wirte_engine(
    hls::stream<int> &slice,
    pkt *core,
    hls::stream<pkt> &core_stream,
    int len
)
{
    int offset = slice.read();
    for(int i = 0; i < len; i++)
    {
#pragma HLS pipeline
        pkt temp = core_stream.read();
        core[offset + i] = temp;
    }
}

void coo_sparser(
    sp_data *data_in,
    hls::stream<int> &indice_0,
    hls::stream<int> &indice_1,
    hls::stream<int> &indice_2,
    hls::stream<int> &indice_3,
    hls::stream<data_t> &data
)
{
#pragma HLS data_pack variable = data_in
#pragma HLS pipeline
    sp_data temp = *data_in;
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

void write_back_engine(
    //memory interface
    pkt *core1,
    pkt *core2,
    pkt *core3,
    pkt *core4,

    hls::stream<pkt> &core1_update,
    hls::stream<pkt> &core2_update,
    hls::stream<pkt> &core3_update,
    hls::stream<pkt> &core4_update,
    hls::stream<sp_data> &update_indices
)
{
    /*
#pragma HLS interface port = core1   mode = m_axi offset = slave bundle = gmem1
#pragma HLS interface port = core2   mode = m_axi offset = slave bundle = gmem2
#pragma HLS interface port = core3   mode = m_axi offset = slave bundle = gmem3
#pragma HLS interface port = core4   mode = m_axi offset = slave bundle = gmem4

#pragma HLS interface s_axilite port = core1
#pragma HLS interface s_axilite port = core2
#pragma HLS interface s_axilite port = core3
#pragma HLS interface s_axilite port = core4
*/
#pragma HLS dataflow

    sp_data temp = update_indices.read();

    int index_0 = temp.indices[0];
    int index_1 = temp.indices[1];
    int index_2 = temp.indices[2];
    int index_3 = temp.indices[3];

    int core1_offset = index_0;
    int core2_offset = index_1 * N;
    int core3_offset = index_2 * N;
    int core4_offset = index_3;

    for(int i = 0; i < 1; i++)
    {
#pragma HLS pipeline
        pkt temp = core1_update.read();
        *(core1 + core1_offset + i) = temp;
    }

    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        pkt temp = core2_update.read();
        *(core2 + core2_offset + i) = temp;
    }

    for(int i = 0; i < N; i++)
    {
#pragma HLS pipeline
        pkt temp = core3_update.read();
        *(core3 + core3_offset + i) = temp;
    }

    for(int i = 0; i < 1; i++)
    {
#pragma HLS pipeline
        pkt temp = core4_update.read();
        *(core4 + core4_offset + i) = temp;
    }
}

void scale(
    data_t scaling,
    hls::stream<pkt> &vec_in,
    hls::stream<pkt> &vec_out
)
{
#pragma HLS dataflow
    data_t vec_buffer[N];
    pkt temp = vec_in.read();
    pkt out;

    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        out.data[i] = temp.data[i] * scaling;
    }

    vec_out.write(out);
}

void update(
    hls::stream<pkt> &core,
    hls::stream<pkt> &grad,
    hls::stream<pkt> &out
)
{   
#pragma HLS pipeline    
    pkt core_ = core.read();
    pkt grad_ = grad.read();
    pkt temp;

    for(int i = 0; i < N; i++)
    {
#pragma HLS unroll
        temp.data[i] = core_.data[i] - grad_.data[i];
    }

    out.write(temp);
}


void pipe(
    sp_data *data_in,
    pkt *core1,
    pkt *core2,
    pkt *core3,
    pkt *core4
)
{
#pragma HLS interface port = data_in mode = m_axi offset = slave bundle = gmem0
#pragma HLS interface port = core1   mode = m_axi offset = slave bundle = gmem1
#pragma HLS interface port = core2   mode = m_axi offset = slave bundle = gmem2
#pragma HLS interface port = core3   mode = m_axi offset = slave bundle = gmem3
#pragma HLS interface port = core4   mode = m_axi offset = slave bundle = gmem4

#pragma HLS interface s_axilite port = data_in
#pragma HLS interface s_axilite port = core1
#pragma HLS interface s_axilite port = core2
#pragma HLS interface s_axilite port = core3
#pragma HLS interface s_axilite port = core4

#pragma HLS dataflow
    hls::stream<pkt> core1_in, core2_in, core3_in, core4_in;
    hls::stream<pkt> core2_pipevm, core2_pipemv;
    hls::stream<pkt> core3_pipevm, core3_pipemv;
    hls::stream<pkt> core1_, core2_, core3_, core4_;
    hls::stream<pkt> core1_for_update, core2_for_update, core3_for_update, core4_for_update;
    //hls::stream<sp_data> data_in_, y, update_indices; 
    hls::stream<pkt> g2l, g3l, g4l, g1r, g2r, g3r;
    hls::stream<pkt> g4l_, g4l__, core4_1, core4_2;
    hls::stream<pkt> grad1, grad2, grad3, grad4;
    hls::stream<pkt> core1_update, core2_update, core3_update, core4_update;
    hls::stream<data_t> x, y;
    hls::stream<int> indice_0, indice_1, indice_2, indice_3, indice_0_, indice_1_, indice_2_, indice_3_;
    hls::stream<int> indice_0_for_update, indice_1_for_update, indice_2_for_update, indice_3_for_update;

    //fetch data
    //data_fetch_engine(data_in, core1, core2, core3, core4, core1_in, core2_in, core3_in, core4_in, data_in_);
    coo_sparser(data_in, indice_0, indice_1, indice_2, indice_3, y);

    dup_stream_int(indice_0, indice_0_, indice_0_for_update);
    dup_stream_int(indice_1, indice_1_, indice_1_for_update);
    dup_stream_int(indice_2, indice_2_, indice_2_for_update);
    dup_stream_int(indice_3, indice_3_, indice_3_for_update);

    read_engine(indice_0_, core1, core1_in, 1);
    read_engine(indice_1_, core2, core2_in, N);
    read_engine(indice_2_, core3, core3_in, N);
    read_engine(indice_3_, core4, core4_in, 1);

    dup_stream(core1_in, core1_, core1_for_update);
    for(int i = 0; i < N; i ++){
        dup_stream(core2_in, core2_, core2_for_update);
        dup_stream(core3_in, core3_, core3_for_update);
    }
    dup_stream(core4_in, core4_, core4_for_update);

    dup_stream(core4_, core4_1, core4_2);
    for(int i = 0; i < N; i++)
    {
        dup_stream(core2_, core2_pipevm, core2_pipemv);
        dup_stream(core3_, core3_pipevm, core3_pipemv);
    }
    //dup_stream_sp(data_in_, y, update_indices);

    //calculate gnl and gnr
    mvchain_pipe(core1_, core2_pipevm, core3_pipevm, core2_pipemv, core3_pipemv, core4_1, g2l, g3l, g4l, g1r, g2r, g3r);

    dup_stream(g4l, g4l_, g4l__);
    
    //calculate x
    dot(g4l_, core4_2, x);
    data_t x_ = x.read();
    data_t y_ = (y.read());
    data_t scaling = (x_ - y_) * 0.01;

    outer(scaling, g2l, g2r, grad2);
    outer(scaling, g3l, g3r, grad3);
    scale(scaling, g1r, grad1);
    scale(scaling, g4l__, grad4);

    update(core1_for_update, grad1, core1_update);
    for(int i = 0; i < N; i++)
    {
        update(core2_for_update, grad2, core2_update);
        update(core3_for_update, grad3, core3_update);
    }
    update(core4_for_update, grad4, core4_update);
    
    //write_back_engine(core1, core2, core3, core4, core1_update, core2_update, core3_update, core4_update, update_indices);
    wirte_engine(indice_0_for_update, core1, core1_update, 1);
    wirte_engine(indice_1_for_update, core2, core2_update, N);
    wirte_engine(indice_2_for_update, core3, core3_update, N);
    wirte_engine(indice_3_for_update, core4, core4_update, 1);
}
