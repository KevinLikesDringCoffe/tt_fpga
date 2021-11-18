TARGET := hw_emu

all : 
	cd ./device && make TARGET=$(TARGET)
	cd ./host && make
	cp ./device/gemv_pipeline.xclbin ./
	cp ./host/host_exe ./

clean :
	rm -rf host_exe gemv_pipeline.xclbin emconfig.json *.log .run .Xil
	cd ./device && make clean
	cd ./host && make clean

run :
	emconfigutil --platform xilinx_u280_xdma_201920_3
	export XCL_EMULATION_MODE=$(TARGET)
	./host_exe -x gemv_pipeline.xclbin

