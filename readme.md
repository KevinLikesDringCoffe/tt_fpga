# Version 0.0 Nov 17
Start from 4.63GB/s poor bandwidth and 3.80x acceleration ratio

How can we do to further optimize it - we should fully use the HBM bandwidth. Synthesis report shows that int current design total 20582 cycles are used. The memory transfer takes 6472 cycles, which is about 33% if the total cycles. Therefore the computation is bounded by the PE, which takes almost all of the cycles. This can be also verified that the current throughput is 1/3 of the ideal HBM bandwidth.  

# Version 0.1 Nov 19
Use streaming MV speed up the gemv to about 13280 cycles. Target is abourt 6000 cycle(to match the memory bandwidth)

WARNING: fadd do not have combination law, therefore there will be difference between the SW and the HW result. For example, assume we are going to calculate

 ```x = a + b + c + d``` 
 
 The software will add them up directly in sequence. However, the hardware may do in this way:

```x1 = a + b; x2 = c + d; x = x1 + x2```

In each addition there will be a loss of accuracy, and HW and SW sometime may get slightly different result.

Here the table shows the result of a hardware acceleratoe of 3-stage gemv chain.

| # of slices | Throughput | Acceleration ratio|
| ----------- | ---------- | ----------------- |
| 1           | 0.0575 GB/s    |  0.0495x          |
| 10          | 0.82 GB/s      |  0.6719x          |
| 100         | 4.8905 GB/s    |  4.0316x          |
| 1000        | 14.294 GB/s    |  11.6882x         |
| 10000       | 18.46 GB/s     |  15.1667x         |


The result of 2-stage gemv chain
| # of slices | Throughput | Acceleration ratio|
| ----------- | ---------- | ----------------- |
| 1           | 0.0473786 GB/s |  0.0419305x       |
| 10          | 0.398993 GB/s  |  0.332374x        |
| 100         | 3.10341 GB/s   |  2.57062x         |
| 1000        | 10.3747 GB/s   |  8.69832x         |
| 10000       | 12.2224 GB/s   |  10.3345x         |

The accelaration ratio acctually benefit linearly from the depth of the pipeline because the pipeline achieves paralleling the computing of each stages of gemv, which in software is caculated sequentially.