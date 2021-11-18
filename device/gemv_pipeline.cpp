#include <hls_stream.h>
#include "gemv32.hpp"

void gemv_pipeline(
    pkt *vec,
    pkt *mat1,
    pkt *mat2,
    pkt *mat3,
    pkt *out
)
{ 
                                                                                                                                                                                                                                              
#pragma HLS interface port=vec  mode=m_axi offset=slave bundle=gmem0
#pragma HLS interface port=mat1 mode=m_axi offset=slave bundle=gmem1
#pragma HLS interface port=mat2 mode=m_axi offset=slave bundle=gmem2
#pragma HLS interface port=mat3 mode=m_axi offset=slave bundle=gmem3
#pragma HLS interface port=out  mode=m_axi offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port = vec
#pragma HLS INTERFACE s_axilite port = mat1
#pragma HLS INTERFACE s_axilite port = mat2
#pragma HLS INTERFACE s_axilite port = mat3
#pragma HLS INTERFACE s_axilite port = out

    hls::stream<pkt> vec_in, mat1_in, mat2_in, mat3_in, out_stream;
    hls::stream<pkt> vec1_2, vec2_3;
    
    int slc = 100;
#pragma HLS dataflow

mat1read:
    for(int i = 0; i < N * BLOCKS * slc; i++)
    {
        pkt temp = mat1[i];
        mat1_in.write(temp);
    }

vecread:
    for(int i = 0; i < BLOCKS * slc; i++)
    {
        pkt temp = vec[i];
        vec_in.write(temp);
    }

pe1:
    for(int i = 0; i < slc; i++)
    {
        gemv_stream(mat1_in, vec_in, vec1_2);
    }

mat2read:
    for(int i = 0; i < N * BLOCKS * slc; i++)
    {
        pkt temp = mat2[i];
        mat2_in.write(temp);
    }

pe2:
    for(int i = 0; i < slc; i++)
    {
        gemv_stream(mat2_in, vec1_2, vec2_3);
    }

mat3read:
    for(int i = 0; i < N * BLOCKS * slc; i++)
    {
        pkt temp = mat3[i];
        mat3_in.write(temp);
    }

pe3:
    for(int i = 0; i < slc; i++)
    {
        gemv_stream(mat3_in, vec2_3, out_stream);
    }


out:
    for(int i = 0; i < BLOCKS * slc; i++)
    {
        pkt temp = out_stream.read();
        out[i] = temp;
    }

}
