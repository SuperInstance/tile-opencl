CC ?= gcc
CL_TARGET_OPENCL_VERSION = 120
CFLAGS = -std=c11 -Wall -Wextra -O2 -Iinclude -DCL_TARGET_OPENCL_VERSION=120
LDFLAGS = -lOpenCL -lm

SRCDIR = src
TESTDIR = tests

SOURCES = $(SRCDIR)/tile_opencl.c
KERNELS = $(SRCDIR)/kernel_hash.cl \
          $(SRCDIR)/kernel_embed.cl \
          $(SRCDIR)/kernel_search.cl \
          $(SRCDIR)/kernel_evolve.cl

.PHONY: all clean test bench

all: tile_test

tile_test: $(SOURCES) $(KERNELS) $(TESTDIR)/test_opencl.c
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(TESTDIR)/test_opencl.c $(LDFLAGS)

test: tile_test
	./tile_test

bench: tile_test
	./tile_test --bench

clean:
	rm -f tile_test
