CLANG ?= clang
UPMEM_HOME ?= ../upmem-2025.1.0-Linux-x86_64
UPMEM_CLANG ?= $(UPMEM_HOME)/bin/dpu-upmem-dpurte-clang
CFLAGS ?= -Wall -Wextra

build: build/llama2.upmem

clean:
	rm -rf build

run: build fetch-models
	UPMEM_PROFILE="backend=simulator" build/llama2.upmem stories15M.bin -s 1 -u

fetch-models:
	curl -fsL -C - -o tokenizer.bin https://github.com/karpathy/llama2.c/raw/refs/heads/master/tokenizer.bin
	curl -fsL -C - -o stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

build/llama2.upmem: build/main.o build/transformer_cpu.o build/transformer_upmem.o
	$(CLANG) build/main.o build/transformer_cpu.o build/transformer_upmem.o -o build/llama2.upmem -L$(UPMEM_HOME)/lib -Wl,-rpath,$(UPMEM_HOME)/lib -lc -lm -ldpu -ldpuverbose

build/main.o: main.c transformer.h
	@mkdir -p $(@D)
	$(CLANG) --std=c11 main.c -c -o build/main.o $(CFLAGS)

build/transformer_cpu.o: transformer.h transformer_cpu.c
	@mkdir -p $(@D)
	$(CLANG) --std=c11 transformer_cpu.c -c -o build/transformer_cpu.o $(CFLAGS)

build/transformer_upmem.o: transformer.h transformer_upmem.c kernels
	@mkdir -p $(@D)
	$(CLANG) --std=c23 -DEMBED_KERNELS transformer_upmem.c -c -o build/transformer_upmem.o -I$(UPMEM_HOME)/include/dpu $(CFLAGS)

kernels: build/attout.kernel build/cls.kernel build/ffn1.kernel build/ffn2.kernel build/mha.kernel build/qkv.kernel build/rmsnorm.kernel build/mha_big.kernel

build/attout.kernel: kernels/attout.c
	@mkdir -p $(@D)
	$(UPMEM_CLANG) -DNR_TASKLETS=16 -o build/attout.kernel kernels/attout.c $(CFLAGS) -O3

build/cls.kernel: kernels/cls.c
	@mkdir -p $(@D)
	$(UPMEM_CLANG) -DNR_TASKLETS=16 -o build/cls.kernel kernels/cls.c $(CFLAGS) -O3

build/ffn1.kernel: kernels/ffn1.c
	@mkdir -p $(@D)
	$(UPMEM_CLANG) -DNR_TASKLETS=16 -o build/ffn1.kernel kernels/ffn1.c $(CFLAGS) -O3

build/ffn2.kernel: kernels/ffn2.c
	@mkdir -p $(@D)
	$(UPMEM_CLANG) -DNR_TASKLETS=16 -o build/ffn2.kernel kernels/ffn2.c $(CFLAGS) -O3

build/mha.kernel: kernels/mha.c
	@mkdir -p $(@D)
	$(UPMEM_CLANG) -DNR_TASKLETS=16 -o build/mha.kernel kernels/mha.c $(CFLAGS) -O3

build/qkv.kernel: kernels/qkv.c
	@mkdir -p $(@D)
	$(UPMEM_CLANG) -DNR_TASKLETS=8 -o build/qkv.kernel kernels/qkv.c $(CFLAGS) -O3

build/rmsnorm.kernel: kernels/rmsnorm.c
	@mkdir -p $(@D)
	$(UPMEM_CLANG) -DNR_TASKLETS=16 -o build/rmsnorm.kernel kernels/rmsnorm.c $(CFLAGS) -O3

build/mha_big.kernel: kernels/mha_big.c
	@mkdir -p $(@D)
	$(UPMEM_CLANG) -DNR_TASKLETS=24 -o build/mha_big.kernel kernels/mha_big.c $(CFLAGS) -O3 -ffast-math
