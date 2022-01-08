TARGET := hw_emu
VPPFLAGS := -g -t $(TARGET) --platform xilinx_u280_xdma_201920_3
PROFILE_FLAGS :=  --profile.data all:all:all --profile.exec all:all --profile.stall all:all
DIRS := 
#DIRS += -I'/opt/Xilinx/Vitis_HLS/2021.1/include'
#DIRS += -I'$(XILINX_VIVADO)/include'
#DIRS += -I'$(XILINX_VITIS)/include'

OBJ := mvpipe_$(TARGET).xo
#OBJ := $(subst .cpp, .xo, $(wildcard *.cpp))
SRC := mvpipe.cpp
DEPS := dup_stream.h tt.h
BIN := mvpipe_$(TARGET).xclbin

$(BIN) : $(OBJ)
	v++ $(VPPFLAGS) $(PROFILE_FLAGS) --link --vivado.impl.jobs 12 --vivado.synth.jobs 12 '$^' -o'$@' --config ./connectivity.cfg

$(OBJ) : $(SRC) $(DEPS)
	v++ $(VPPFLAGS) --hls.pre_tcl ./runPre.tcl -c -k pipe $(DIRS) -I'$(<D)' -o'$@' $<

.PHONY : clean run

clean :
	rm -rf _x *.log *.compile_summary *.xo .ipcache .Xil *.xclbin *.ltx *.info *.link_summary

