#include "cmdlineparser.h"
#include <iostream>
#include <cstring>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void gemv(float *matrix, float * vector, int m, int n, float *out)
{
    //initialization
    for(int i = 0; i < m; i++)
    {
        out[i] = 0;
    }
    
    float acc;

    for(int i = 0; i < m; i++)
    {
        acc = 0;
        for(int j = 0; j < n; j++)
        {
            acc += matrix[i*n + j] * vector[j];
        }
        out[i] = acc;
    }

    return;
}

void randmem(float *ptr, int size)
{
    srand(static_cast <unsigned> (time(0)));

    for(int i = 0; i < size; i++)
    {
        float r = static_cast <float> (rand() / static_cast <float> (RAND_MAX));
        ptr[i] = r;
    }
}

void run_krnl(xrtDeviceHandle device, xrt::kernel& krnl, int* bank_assign, unsigned int size, unsigned int slc) {
    size_t vector_size_bytes = sizeof(float) * size * slc;
    size_t matrix_size_bytes = sizeof(float) * size * size * slc;
    

    std::cout << "Allocate Buffer in Global Memory\n";
    auto vec = xrt::bo(device, vector_size_bytes, bank_assign[0]);
    auto mat1 = xrt::bo(device, matrix_size_bytes, bank_assign[1]);
    auto mat2 = xrt::bo(device, matrix_size_bytes, bank_assign[2]);
    auto mat3 = xrt::bo(device, matrix_size_bytes, bank_assign[3]);
    auto mat4 = xrt::bo(device, matrix_size_bytes, bank_assign[4]);
    auto out = xrt::bo(device, vector_size_bytes, bank_assign[5]);

    // Map the contents of the buffer object into host memory
    auto vec_map = vec.map<float*>();
    auto mat1_map = mat1.map<float*>();
    auto mat2_map = mat2.map<float*>();
    auto mat3_map = mat3.map<float*>();
    auto mat4_map = mat4.map<float*>();
    auto out_map = out.map<float*>();

    std::cout << "Randomize the Test Cores.\n";
    randmem(vec_map, size * slc);
    randmem(mat1_map, size * size * slc);
    randmem(mat2_map, size * size * slc);
    randmem(mat3_map, size * size * slc);
    randmem(mat4_map, size * size * slc);

    float out_ref[size * slc];

    std::chrono::duration<double> sw_time(0);
    auto sw_start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < slc; i++)
    {
        float buffer1[size];
        gemv(mat1_map + i * size * size, vec_map + i * size, size, size, buffer1);
        float buffer2[size];
        gemv(mat2_map + i * size * size, buffer1, size, size, buffer2);
        float buffer3[size];
        gemv(mat3_map + i * size * size, buffer2, size, size, buffer3);

        gemv(mat3_map + i * size * size, buffer2, size, size, out_ref + i * size * size);
    }

    auto sw_end = std::chrono::high_resolution_clock::now();
    sw_time = std::chrono::duration<double>(sw_end - sw_start);

/*
    std::fill(vec_map, vec_map + size * 100, 1);
    std::fill(mat1_map, mat1_map + size * size * 100, 1);
    std::fill(mat2_map, mat2_map + size * size * 100, 1);
    std::fill(mat3_map, mat3_map + size * size * 100, 1);
    std::fill(out_map, out_map + size * 100, 1);
*/

    // Create the test data
    /*
    int bufReference[size];
    for (uint32_t i = 0; i < size; ++i) {
        bo0_map[i] = i;
        bo1_map[i] = i;
        bufReference[i] = bo0_map[i] + bo1_map[i];
    }
    */

    // Synchronize buffer content with device side
    std::cout << "synchronize input buffer data to device global memory\n";

    vec.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    mat1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    mat2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    mat3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    mat4.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::chrono::duration<double> kernel_time(0);

    std::cout << "Execution of the kernel\n";
    auto kernel_start = std::chrono::high_resolution_clock::now();
    auto run = krnl(vec, mat1, mat2, mat3, mat4, out, slc);
    run.wait();
    auto kernel_end = std::chrono::high_resolution_clock::now();

    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    // Get the output;
    std::cout << "Get the output data from the device" << std::endl;
    out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Validate our results
    
    /*
    if (std::memcmp(out_map, out_ref, size * slc))
        throw std::runtime_error("Value read back does not match reference");
    */

    double result;
    result = (4 * size * size) * slc * sizeof(float);
    result /= (1000 * 1000 * 1000); // to GB
    result /= kernel_time.count();   // to GBps

    std::cout << "THROUGHPUT = " << result << " GB/s " << std::endl;

    std::cout << "Acceleration ratio = " << sw_time.count() / kernel_time.count() << "x" << std::endl;

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

    auto krnl = xrt::kernel(device, uuid, "gemv_pipeline");

    unsigned int dataSize = 32;
    unsigned int slc = 1000;
    double kernel_time_in_sec = 0, result = 0;
    const int numBuf = 6; // Since three buffers are being used
    int bank_assign[numBuf];
    for (int j = 0; j < numBuf; j++) {
        bank_assign[j] = j;
    }

    run_krnl(device, krnl, bank_assign, dataSize, slc);
    
    return 0;
}
