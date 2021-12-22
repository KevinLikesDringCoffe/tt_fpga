cc = gcc
target = tt_sgd
obj = tt_sgd.o update_slice.o get_idx.o dot.o scale.o sumup.o \
outer.o g_nl.o g_nr.o gemv.o gevm.o core2tensor.o rand_tensor.o\
rand_core.o ones_tensor.o rand_sample.o sgd_engine.o rand_sample_sp_data.o
deps = tt_sgd.h

$(target) : $(obj)
	$(cc) -o $(target) $(obj)


%.o : %.c $(deps)
	$(cc) -o $@ -c $<

.PHONY: clean

clean: 
	rm -rf $(obj) $(target)