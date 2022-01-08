template <typename T>
void dup_stream(
    hls::stream<T> &in,
    hls::stream<T> &dup1,
    hls::stream<T> &dup2,
    int len
)
{
    for(int i = 0 ; i < len; i++)
    {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 16 max = 16
        T temp = in.read();
        dup1.write(temp);
        dup2.write(temp);
    }
}