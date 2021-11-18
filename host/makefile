EXE := ./host_exe
HOST_SRC = ./includes/cmdparser/cmdlineparser.cpp ./includes/logger/logger.cpp ./host.cpp
CPPFLAGS := -I$(XILINX_XRT)/include -L$(XILINX_XRT)/lib -I$(XILINX_XRT)/include/experimental
CPPFLAGS += -I./includes/cmdparser
CPPFLAGS += -I./includes/logger
LDFLAGS := -std=c++14
LDFLAGS += -lxrt_coreutil -pthread
LDFLAGS += -g

$(EXE) : $(HOST_SRC) 
	g++ $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY : clean

clean : 
	rm host_exe