CC=gcc
CFLAGS=-Wall -g -std=c99
LDFLAGS=
LDLIBS=-lm

# For LAPACKE (C interface to LAPACK)
LAPACKE_CFLAGS_DETECTED=$(shell pkg-config --cflags lapacke 2>/dev/null || echo "")
LAPACKE_LDLIBS_DETECTED=$(shell pkg-config --libs lapacke 2>/dev/null || echo "")

# For GSL (GNU Scientific Library)
GSL_CFLAGS_DETECTED=$(shell gsl-config --cflags 2>/dev/null || echo "")
GSL_LDLIBS_DETECTED=$(shell gsl-config --libs 2>/dev/null || echo "")

# For this subtask, we know GSL was installed and LAPACKE was not.
# So, we will explicitly use GSL flags.
# A more dynamic Makefile could be made later.
FINAL_CFLAGS=$(GSL_CFLAGS_DETECTED)
FINAL_LDLIBS=$(GSL_LDLIBS_DETECTED)

# Fallback if gsl-config also somehow failed (should not happen based on previous step)
ifeq ($(strip $(GSL_CFLAGS_DETECTED)),)
    $(warning "GSL CFLAGS not found, compilation might fail or miss headers.")
endif
ifeq ($(strip $(GSL_LDLIBS_DETECTED)),)
    $(warning "GSL LDLIBS not found, linking might fail.")
    # Add default GSL libs as a last resort if gsl-config failed but libs might be in standard paths
    # FINAL_LDLIBS += -lgsl -lgslcblas
endif


TARGET=pca_demo

SRCS=main.c pca.c
OBJS=$(SRCS:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) $(FINAL_LDLIBS) -o $@

%.o: %.c pca.h
	$(CC) $(CFLAGS) $(FINAL_CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
