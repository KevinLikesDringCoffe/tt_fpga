TARGET := hw_emu

all : 
	cd ./device && make TARGET=$(TARGET)
	cd ./host && make
	cp ./device/mvpipe_$(TARGET).xclbin ./
	cp ./host/host_exe ./

clean :
	rm -rf host_exe *.xclbin *.log .run .Xil *.csv *.run_summary *.wdb *.wcfg *.protoinst
	cd ./device && make clean
	cd ./host && make clean

run :
	emconfigutil --platform xilinx_u280_xdma_201920_3
	export XCL_EMULATION_MODE=$(TARGET)
	./host_exe -x mvpipe_$(TARGET).xclbin

