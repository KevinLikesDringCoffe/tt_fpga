#include <hls_stream.h>
#include "tt.h"

//extern "C"{
void pipe(
    sp_data *data_in,
    pkt *core1,
    pkt *core2,
    pkt *core3,
    pkt *core4,
    int slc
)
{
#pragma HLS interface port = data_in mode = m_axi offset = slave bundle = gmem0 latency = 0 num_read_outstanding=32 num_write_outstanding=32 
#pragma HLS interface port = core1   mode = m_axi offset = slave bundle = gmem1 latency = 0 num_read_outstanding=32 num_write_outstanding=32 
#pragma HLS interface port = core2   mode = m_axi offset = slave bundle = gmem2 latency = 0 num_read_outstanding=32 num_write_outstanding=32 
#pragma HLS interface port = core3   mode = m_axi offset = slave bundle = gmem3 latency = 0 num_read_outstanding=32 num_write_outstanding=32 
#pragma HLS interface port = core4   mode = m_axi offset = slave bundle = gmem4 latency = 0 num_read_outstanding=32 num_write_outstanding=32

#pragma HLS interface s_axilite port = data_in
#pragma HLS interface s_axilite port = core1
#pragma HLS interface s_axilite port = core2
#pragma HLS interface s_axilite port = core3
#pragma HLS interface s_axilite port = core4
#pragma HLS interface s_axilite port = slc
    
#pragma HLS dataflow

    hls::stream<pkt> core1_in, core2_in, core3_in, core4_in;
    hls::stream<pkt> core2_pipevm, core2_pipemv;
    hls::stream<pkt> core3_pipevm, core3_pipemv;
    hls::stream<pkt> core1_, core2_, core3_, core4_;
    hls::stream<pkt> core1_for_update, core2_for_update, core3_for_update, core4_for_update;
    hls::stream<pkt> g2l, g3l, g4l, g1r, g2r, g3r;
    hls::stream<pkt> g4l_, g4l__, core4_1, core4_2;
    hls::stream<pkt> grad1, grad2, grad3, grad4;
    hls::stream<pkt> core1_update, core2_update, core3_update, core4_update;
    hls::stream<data_t> x, y;
    hls::stream<int> indice_0, indice_1, indice_2, indice_3, indice_0_, indice_1_, indice_2_, indice_3_;
    hls::stream<int> indice_0_for_update, indice_1_for_update, indice_2_for_update, indice_3_for_update;

    //fetch data
#pragma HLS stream variable=core1_in type=fifo depth=16 
#pragma HLS stream variable=core2_in type=fifo depth=16 
#pragma HLS stream variable=core3_in type=fifo depth=16 
#pragma HLS stream variable=core4_in type=fifo depth=16 

#pragma HLS stream variable=core2_pipevm type=fifo depth=16
#pragma HLS stream variable=core2_pipemv type=fifo depth=16 

#pragma HLS stream variable=core3_pipevm type=fifo depth=16
#pragma HLS stream variable=core3_pipemv type=fifo depth=16 

#pragma HLS stream variable=core1_ type=fifo depth=16 
#pragma HLS stream variable=core2_ type=fifo depth=16 
#pragma HLS stream variable=core3_ type=fifo depth=16 
#pragma HLS stream variable=core4_ type=fifo depth=16 

#pragma HLS stream variable=core1_for_update type=fifo depth=16 
#pragma HLS stream variable=core2_for_update type=fifo depth=16 
#pragma HLS stream variable=core3_for_update type=fifo depth=16 
#pragma HLS stream variable=core4_for_update type=fifo depth=16 

#pragma HLS stream variable=g2l type=fifo depth=16 
#pragma HLS stream variable=g3l type=fifo depth=16 
#pragma HLS stream variable=g4l type=fifo depth=16 
#pragma HLS stream variable=g1r type=fifo depth=16 
#pragma HLS stream variable=g2r type=fifo depth=16 
#pragma HLS stream variable=g3r type=fifo depth=16 

#pragma HLS stream variable=g4l_ type=fifo depth=16 
#pragma HLS stream variable=g4l__ type=fifo depth=16 
#pragma HLS stream variable=core4_1 type=fifo depth=16 
#pragma HLS stream variable=core4_2 type=fifo depth=16

#pragma HLS stream variable=grad1 type=fifo depth=16 
#pragma HLS stream variable=grad2 type=fifo depth=16 
#pragma HLS stream variable=grad3 type=fifo depth=16 
#pragma HLS stream variable=grad4 type=fifo depth=16

#pragma HLS stream variable=core1_update type=fifo depth=16 
#pragma HLS stream variable=core2_update type=fifo depth=16 
#pragma HLS stream variable=core3_update type=fifo depth=16 
#pragma HLS stream variable=core4_update type=fifo depth=16 

#pragma HLS stream variable=x type=fifo depth=16 
#pragma HLS stream variable=y type=fifo depth=16

#pragma HLS stream variable=indice_0 type=fifo depth=16 
#pragma HLS stream variable=indice_1 type=fifo depth=16 
#pragma HLS stream variable=indice_2 type=fifo depth=16 
#pragma HLS stream variable=indice_3 type=fifo depth=16 
#pragma HLS stream variable=indice_0_ type=fifo depth=16 
#pragma HLS stream variable=indice_1_ type=fifo depth=16 
#pragma HLS stream variable=indice_2_ type=fifo depth=16 
#pragma HLS stream variable=indice_3_ type=fifo depth=16 

#pragma HLS stream variable=indice_0_for_update type=fifo depth=16 
#pragma HLS stream variable=indice_1_for_update type=fifo depth=16 
#pragma HLS stream variable=indice_2_for_update type=fifo depth=16 
#pragma HLS stream variable=indice_3_for_update type=fifo depth=16 

coo_read:
    for(int iter = 0; iter < slc; iter++)
    {
#pragma HLS loop_tripcount min = 100 max = 1000000
        coo_sparser(data_in, indice_0, indice_1, indice_2, indice_3, y, iter);
    }

index_dup:
    for(int iter = 0; iter < slc; iter++)
    {
#pragma HLS loop_tripcount min = 100 max = 1000000
        dup_stream<int>(indice_0, indice_0_, indice_0_for_update, 1);
        dup_stream<int>(indice_1, indice_1_, indice_1_for_update, 1);
        dup_stream<int>(indice_2, indice_2_, indice_2_for_update, 1);
        dup_stream<int>(indice_3, indice_3_, indice_3_for_update, 1);
    }

core_read:
    for(int iter = 0; iter < slc; iter++)
    {
#pragma HLS loop_tripcount min = 100 max = 1000000
        read_engine(indice_0_, core1, core1_in, 1);
        read_engine(indice_1_, core2, core2_in, N);
        read_engine(indice_2_, core3, core3_in, N);
        read_engine(indice_3_, core4, core4_in, 1);

    }

core_dup:
    for(int iter = 0; iter < slc; iter++)
    {
#pragma HLS loop_tripcount min = 100 max = 1000000
        dup_stream<pkt>(core1_in, core1_, core1_for_update, 1);
        dup_stream<pkt>(core2_in, core2_, core2_for_update, N);
        dup_stream<pkt>(core3_in, core3_, core3_for_update, N);
        dup_stream<pkt>(core4_in, core4_, core4_for_update, 1);

        dup_stream<pkt>(core4_, core4_1, core4_2, 1);
        dup_stream<pkt>(core2_, core2_pipevm, core2_pipemv, N);
        dup_stream<pkt>(core3_, core3_pipevm, core3_pipemv, N);
    }
        //dup_stream_sp(data_in_, y, update_indices);

        //calculate gnl and gnr
mv_chain:
    for(int iter = 0; iter < slc; iter++)
    {
#pragma HLS loop_tripcount min = 100 max = 1000000
        mvchain_pipe(core1_, core2_pipevm, core3_pipevm, core2_pipemv, core3_pipemv, core4_1, g2l, g3l, g4l, g1r, g2r, g3r);
    }

stream_dup:
    for(int iter = 0; iter < slc; iter++)
    {
#pragma HLS loop_tripcount min = 100 max = 1000000
        dup_stream<pkt>(g4l, g4l_, g4l__, 1);
    }    
        //calculate x

outer:
    for(int iter = 0; iter < slc; iter++)
    {
#pragma HLS loop_tripcount min = 100 max = 1000000
        dot(g4l_, core4_2, x);
        data_t x_ = x.read();
        data_t y_ = y.read();
        data_t scaling = (x_ - y_) * 0.00001;

        outer(scaling, g2l, g2r, grad2);
        outer(scaling, g3l, g3r, grad3);
        scale(scaling, g1r, grad1);
        scale(scaling, g4l__, grad4);

        //loss += (x_ - y_) * (x_ - y_);
    }

update:
    for(int iter = 0; iter < slc; iter++)
    {
#pragma HLS loop_tripcount min = 100 max = 1000000
        update(core1_for_update, grad1, core1_update, 1);
        update(core2_for_update, grad2, core2_update, N);
        update(core3_for_update, grad3, core3_update, N);
        update(core4_for_update, grad4, core4_update, 1);
    }

        //write_back_engine(core1, core2, core3, core4, core1_update, core2_update, core3_update, core4_update, update_indices);
write_back:   
    for(int iter = 0; iter < slc; iter++){
#pragma HLS loop_tripcount min = 100 max = 1000000
        write_engine(indice_0_for_update, core1, core1_update, 1);
        write_engine(indice_1_for_update, core2, core2_update, N);
        write_engine(indice_2_for_update, core3, core3_update, N);
        write_engine(indice_3_for_update, core4, core4_update, 1);
    }
}
//}