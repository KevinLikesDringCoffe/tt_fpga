TARGET := hw_emu
VPPFLAGS := -t $(TARGET) --platform xilinx_u280_xdma_201920_3
DIRS := 
#DIRS += -I'/opt/Xilinx/Vitis_HLS/2021.1/include'
#DIRS += -I'$(XILINX_VIVADO)/include'
#DIRS += -I'$(XILINX_VITIS)/include'

OBJ := mvpipe.xo
#OBJ := $(subst .cpp, .xo, $(wildcard *.cpp))
DEPS := 
BIN := mvpipe.xclbin

$(BIN) : $(OBJ)
	v++ $(VPPFLAGS) --link --vivado.impl.jobs 12 --vivado.synth.jobs 12 '$^' -o'$@' --config ./connectivity.cfg

%.xo : %.cpp $(DEPS)
	v++ $(VPPFLAGS) -c -k pipe $(DIRS) -I'$(<D)' -o'$@' $<

.PHONY : clean run

clean :
	rm -rf _x *.log *.compile_summary *.xo .ipcache .Xil *.xclbin *.ltx *.info *.link_summary

