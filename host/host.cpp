#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include "tt_sgd.h"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define M 4

void run_krnl(xrtDeviceHandle device, xrt::kernel& krnl, int* bank_assign, int *tensor_size) {
    float mr = 0.8;
    float margin = 0.05;
    int mode = M;
    int tt_rank[M + 1] = {1, 16, 16, 16, 1};

    float *tt_core[mode];
    float *grad[mode];

    int len = 1;

    for(int i = 0; i < mode; i++)
    {
        len *= tensor_size[i];
    }

    float *t = (float *) malloc(len * sizeof(float));

    ones_tensor(tensor_size, mode, t);

    //allocate space for the tt core and initialization
    /*
    for(int i = 0; i < mode; i++)
    {
        tt_core[i] = (float *) malloc(tt_rank[i] * tt_rank[i+1] * tensor_size[i] * sizeof(float));
        rand_core(tt_rank[i], tt_rank[i+1], tensor_size[i], tt_core[i]);
    }
    */

    //allocate space for recovered tensor
    float *out = (float *) malloc(len * sizeof(float));

    std::cout << "Allocate Buffer in Global Memory\n";
    auto sp_in = xrt::bo(device, (int) (sizeof(sp_data) * len * (mr + margin)), krnl.group_id(0));
    auto core1 = xrt::bo(device, tt_rank[0] * tt_rank[1] * tensor_size[0] * sizeof(float), krnl.group_id(1));
    auto core2 = xrt::bo(device, tt_rank[1] * tt_rank[2] * tensor_size[1] * sizeof(float), krnl.group_id(2));
    auto core3 = xrt::bo(device, tt_rank[2] * tt_rank[3] * tensor_size[2] * sizeof(float), krnl.group_id(3));
    auto core4 = xrt::bo(device, tt_rank[3] * tt_rank[4] * tensor_size[3] * sizeof(float), krnl.group_id(4));
    
    std::cout << "The memory bank of the correspoding arguments are : "<< krnl.group_id(0) <<krnl.group_id(1) << krnl.group_id(2) << krnl.group_id(3) << krnl.group_id(4) << std::endl;
    // Map the contents of the buffer object into host memory
    auto sp_in_map = sp_in.map<sp_data*>();
    auto core1_map = core1.map<float*>();
    auto core2_map = core2.map<float*>();
    auto core3_map = core3.map<float*>();
    auto core4_map = core4.map<float*>();

    std::cout << "Randomize the TT Cores.\n";
    rand_core(tt_rank[0], tt_rank[1], tensor_size[0], core1_map);
    rand_core(tt_rank[1], tt_rank[2], tensor_size[1], core2_map);
    rand_core(tt_rank[2], tt_rank[3], tensor_size[2], core3_map);
    rand_core(tt_rank[3], tt_rank[4], tensor_size[3], core4_map);
    int nnz = rand_sample_sp_data(t, mode, tensor_size, mr, sp_in_map);

    tt_core[0] = core1_map;
    tt_core[1] = core2_map;
    tt_core[2] = core3_map;
    tt_core[3] = core4_map;
/*
    std::cout << "Start sw emulation" << std::endl;

    std::chrono::duration<double> host_time(0);

    std::cout << "Execution of the kernel\n";

    //auto host_start = std::chrono::high_resolution_clock::now();
    sgd_engine(sp_in_map, nnz, mode, tt_rank, tensor_size, tt_core, out, 0.000001, 1);

    //auto host_end = std::chrono::high_resolution_clock::now();
    //host_time = std::chrono::duration<double>(host_end - host_start);

    //std::cout << "The SW execution time is" << host_time.count() << std::endl;
    // Get the output;
*/
   
    std::cout << "Synchronize input buffer data to device global memory\n";

    sp_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    core1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    core2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    core3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    core4.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::chrono::duration<double> kernel_time(0);

    std::cout << "Execution of the kernel\n";
    auto kernel_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 50; i++)
    {
        auto run = krnl(sp_in, core1, core2, core3, core4, nnz);
        run.wait();
    }
    
    auto kernel_end = std::chrono::high_resolution_clock::now();

    

    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    // Get the output;
    std::cout << "Get the output data from the device" << std::endl;
    core1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    core2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    core3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    core4.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    core2tensor(tt_core, tt_rank, tensor_size, mode, out);

    for(int i = 0; i < 100; i++)
    {
        std::cout << out[i] << " ";
    }
    // Validate our results
    
    /*
    if (std::memcmp(out_map, out_ref, size * slc))
        throw std::runtime_error("Value read back does not match reference");
    */

    double result;
    result = ((2 + 2/16 + 10/512) * 16 * 16)  * sizeof(float) * nnz;
    result /= (1000 * 1000 * 1000); // to GB
    result /= kernel_time.count();   // to GBps

    std::cout << std::endl << "THE TOTAL KERNEL TIME COST IS " << kernel_time.count() << std::endl;
    std::cout << "THROUGHPUT = " << result << " GB/s " << std::endl;

    //std::cout << "Acceleration ratio = " << sw_time.count() / kernel_time.count() << "x" << std::endl;

    std::cout << "TEST PASSED" << std::endl;

    return;
}

int main(int argc, char* argv[]) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));

    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    auto krnl = xrt::kernel(device, uuid, "pipe");
    const int numBuf = M + 1; // Since three buffers are being used

    int bank_assign[numBuf];
    for (int j = 0; j < numBuf; j++) {
        bank_assign[j] = j;
    }

    int size[M] = {50, 50, 50, 50};

    run_krnl(device, krnl, bank_assign, size);
    
    return 0;
}
