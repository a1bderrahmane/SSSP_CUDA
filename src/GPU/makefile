NVCC = nvcc
NVCC_FLAGS = -std=c++17 -Xcompiler -Wall -O3
CUDA_ARCH = -arch=native


CC = g++
CFLAGS = -std=c++17 -Wall -O3

TARGET = main_cuda

# Define the source files
CU_SOURCES = main.cu GPUsolver.cu
CPP_SOURCES = CSR.cpp
HEADER_FILES = CSR.hpp GPUsolver.cuh utils.cuh ../Isolver.cuh

# Define object files
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -c $< -o $@


%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)