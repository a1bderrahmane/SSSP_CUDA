# ==========================================
# CUDA SSSP Project Makefile
# ==========================================
NVCC        := nvcc
CXX         := g++

ARCH        := -arch=sm_75

COMMON_FLAGS := -O3
NVCC_FLAGS   := $(ARCH) $(COMMON_FLAGS) --compiler-options "-Wall -Wextra"
LDFLAGS      := $(ARCH)
CPU_LDLIBS   := -pthread

INCLUDES    := -I./src -I./src/CPU -I./src/GPU -I./src/Hybrid

SRC_DIR     := src
OBJ_DIR     := obj
BIN_DIR     := exec

CPU_TARGET  := $(BIN_DIR)/cpu_solver
GPU_TARGET  := $(BIN_DIR)/gpu_solver

SHARED_CPP_SRCS := $(SRC_DIR)/CSR.cpp
CPU_CPP_SRCS    := $(wildcard $(SRC_DIR)/CPU/*.cpp)
GPU_CU_SRCS     := $(wildcard $(SRC_DIR)/GPU/*.cu) \
                   $(wildcard $(SRC_DIR)/Hybrid/*.cu) \
                   $(wildcard $(SRC_DIR)/*.cu)
GPU_CU_SRCS     := $(sort $(GPU_CU_SRCS))

SHARED_OBJS := $(SHARED_CPP_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CPU_OBJS    := $(CPU_CPP_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
GPU_OBJS    := $(GPU_CU_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

all: cpu

cpu: $(CPU_TARGET)

gpu: cpu $(GPU_TARGET)

$(CPU_TARGET): $(SHARED_OBJS) $(CPU_OBJS) | $(BIN_DIR)
	@echo "Linking CPU solver -> $@"
	$(CXX) $(COMMON_FLAGS) $(SHARED_OBJS) $(CPU_OBJS) $(CPU_LDLIBS) -o $@

$(GPU_TARGET): $(SHARED_OBJS) $(GPU_OBJS) | $(BIN_DIR)
	@echo "Linking GPU solver -> $@"
	$(NVCC) $(LDFLAGS) $(SHARED_OBJS) $(GPU_OBJS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling C++: $<"
	$(CXX) $(INCLUDES) $(COMMON_FLAGS) -std=c++14 -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(INCLUDES) $(NVCC_FLAGS) -dc $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR)
	rm -f $(CPU_TARGET) $(GPU_TARGET)

info:
	@echo "CPU C++ Sources: $(CPU_CPP_SRCS)"
	@echo "Shared C++ Sources: $(SHARED_CPP_SRCS)"
	@echo "GPU CUDA Sources: $(GPU_CU_SRCS)"

.PHONY: all clean info cpu gpu
