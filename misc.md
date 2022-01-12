# 关于HLS的一些记录

kernel有的时候会卡死，如果ctrl+c结束kernnel有的时候会出现问题，再次启动kernel的时候会出现找不到板卡的错误。此时可以尝试reset板卡，如果reset也不行的话只能reboot机器解决了。所以建议测试顺序是先过hls的simulation， 然后是sw_emu，hw_emu，最后再是hw。不然一上来卡死了就可能要reboot，很麻烦。

一些reference: 
[1.UG1393](https://docs.xilinx.com/r/zh-CN/ug1393-vitis-application-acceleration/%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E5%BA%94%E7%94%A8%E5%8A%A0%E9%80%9F%E5%BC%80%E5%8F%91%E6%B5%81%E7%A8%8B)

![hls 优化流程](https://docs.xilinx.com/api/khub/maps/e0yeNYHBEoTuQopmlaaYEg/resources/mTaAhfbDYc_XLR2gbnAoFQ/content?Ft-Calling-App=ft%2Fturnkey-portal&Ft-Calling-App-Version=3.10.38&nocache=1641355902470)

## 各种emulation
### SW emulation
SW emulation的时候一直会报错。可能需要在模块里加上```template <int DUMMY = 0>```。但是再HLS工具自带的simulation里面仿真的时候函数是不能有这个的，不然回报找不到对应function的错误。
但是加上这句之后会显示找不到相应的function，不加又会出现kernel name doesnot match。总之这个问题还不知道怎么解决。【5 min 之后】已解决，解决方式是要在顶层函数外侧加上```extern "C"```

## tt_fpga项目相关
之前kernel虽然过了HLS工具自带的simulation，但是一上hw就一直卡死，于是打算从sw_emu和hw_emu开始做，结果hw_emu中kernel也是卡死。今天刚刚把sw_emu跑通，发现sw_emu会直接中断并报错，报的是```No compute units satisfy requested connectivity```也许是connectivity的问题。de了差不多一个小时的bug，最后发现如果在```xrt::bo```开辟memory的时候用bank_assign就会出现问题，但是用```krnl.group_id```就好了。很奇怪，我打算print一下```krnl.group_id```都是多少。理论上我用HBM的话应该是0-5. 好家伙，一看发现全是0，意思是5个arguments全都分配到了HBM bank 0.但是我在connectivity.cfg里面不是这样写的，是分别分配了memory bank。好像再sw_emu里面所有的args都会分到bank0，但是在hw_emu里面就会有区别。现在sw_emu过了，但是hw_emu依然卡死。估计是fifo深度不够的问题。这个我明天再想一下。HLS综合出来的fifo好像深度只有3.把所有的fifo depth强行改成了10.看看还有没有这个问题.depth改成10不太行，但是改成100好像可以了。所以到底是多少才行呢?后来的测试是改为16深度就可以。

### 1.10 记录
目前的版本大概是做了一些优化，相较于之前版本是设置了config_compile -unsafe_math_optimizations.这个设置将会使累加的代码综合出来成为树状的加法器。从而减少cycles。目前完成50*50*50*50张量补全的时间大概为1s左右，throughput为2.51GB/s相比于理论的29GB，不到十分之一。

### 1.11 一些相关想法
对于目前设计的优化可以从两个方面去考虑：硬件和软件。当然software与hardware的co-design是少不了的，但是为了简化现在的考虑，所以暂时分开。目前的设计没有加入raw-solver，所以优化算法并不是真正的sgd，而是弱化版的mini-batch gradient descent。这样的设计最后可以收敛，但是收敛速度要比sgd慢很多。所以虽然一个iteration是快了，但是考虑到实际效果，其实是慢了。因此之后的该进可以分为硬件和软件（算法）两个方面。首先在算法上，1）需要支持各种各样的优化器。这些优化器有的可能需要历史梯度信息。而这些梯度信息可能需要开辟新的内存去存储。2）其次，coo数据格式可能需要支持分块，以支持并行的CU设计。在硬件上1）需要添加raw-solver使之支持真正的sgd算法2）与1）相结合，每次读取可以不用从memory读，而是可以从片上的cache或是buffer内读取，这样可以快一些。

### 1.12 记录
将一晚上编译出来的kernel进行了一个测试。改进之处是将m_axi_latency从64改为了0.这一举措将本来的吞吐量从2.5提升到了2.6，效果不是很大。因此我估计目前的memory bandwidth并不是系统的瓶颈。

## 关于Vitis Kernel Flow和Vivado IP Flow的区别
一个memread的模块，两种Flow综合出来的cycle会有很大区别。这可能是由于两个Flow下面axi burst的配置不同。但是测试用pragma HLS更改latency实际上并不会对最终的cycle产生多大的影响。