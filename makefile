# ==========================================
# CUDA SSSP Project Makefile
# ==========================================
NVCC        := nvcc
CXX         := g++

ARCH        := -arch=sm_75

COMMON_FLAGS := -O3
NVCC_FLAGS   := $(ARCH) $(COMMON_FLAGS) --compiler-options "-Wall -Wextra"
LDFLAGS      := $(ARCH)

INCLUDES    := -I./src -I./src/CPU -I./src/GPU -I./src/Hybrid

SRC_DIR     := src
OBJ_DIR     := obj
BIN_DIR     := exec
TARGET      := $(BIN_DIR)/sssp_solver


ALL_CPP     := $(shell find $(SRC_DIR) -name "*.cpp")
ALL_CU      := $(shell find $(SRC_DIR) -name "*.cu")

EXCLUDE_CU  := $(SRC_DIR)/GPU/main.cu

SRCS_CU     := $(filter-out $(EXCLUDE_CU), $(ALL_CU))
SRCS_CPP    := $(ALL_CPP)

OBJS        := $(SRCS_CPP:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o) \
               $(SRCS_CU:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)


all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	@echo "Linking $@"
	$(NVCC) $(LDFLAGS) $(OBJS) -o $@

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
	rm -f $(TARGET)

info:
	@echo "CUDA Sources: $(SRCS_CU)"
	@echo "C++ Sources:  $(SRCS_CPP)"

.PHONY: all clean info