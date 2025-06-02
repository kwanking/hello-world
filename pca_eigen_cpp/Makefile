# Compiler
CXX=g++
# Compiler flags
CXXFLAGS=-Wall -g -std=c++11 # Or -std=c++14, -std=c++17 etc.
# Include flags for Eigen
# First, try pkg-config if available (common for system-wide installs)
EIGEN_CFLAGS_PKGCONFIG=$(shell pkg-config --cflags eigen3 2>/dev/null)
# If pkg-config fails or returns nothing, try common default include path
ifeq ($(strip $(EIGEN_CFLAGS_PKGCONFIG)),)
    EIGEN_CFLAGS=-I/usr/include/eigen3
else
    EIGEN_CFLAGS=$(EIGEN_CFLAGS_PKGCONFIG)
endif

# Target executable
TARGET=pca_eigen_demo

# Source files
SRCS=main.cpp pca_eigen.cpp
# Object files
OBJS=$(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

%.o: %.cpp pca_eigen.h
	$(CXX) $(CXXFLAGS) $(EIGEN_CFLAGS) -c $< -o $@

# Rule for main.o which might not need pca_eigen.h directly but is good practice
# This explicit rule for main.o might be redundant if the generic %.o rule is sufficient
# and main.cpp includes pca_eigen.h (even if empty).
# However, to be safe and ensure EIGEN_CFLAGS are applied if main.cpp uses Eigen types directly later:
main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CFLAGS) -c main.cpp -o main.o


clean:
	rm -f $(OBJS) $(TARGET)
