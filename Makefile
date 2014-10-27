CC=g++
CFLAGS=-DNDEBUG -O3 -Wall -IDBoW2 \
  $(shell pkg-config --cflags opencv)
LFLAGS=-lDBoW2 -lDVision -lDUtilsCV -lDUtils \
  $(shell pkg-config --libs opencv) -lstdc++

DEPS=lib/libDBoW2.so
TARGET=demo

all: $(TARGET) 

$(TARGET): $(TARGET).o $(DEPS)
	$(CC) $(TARGET).o $(LFLAGS) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ 

lib/libDBoW2.so:
	make -C DBoW2 && mkdir -p ./lib/ && cp DBoW2/libDBoW2.so ./lib/

clean:
	rm -f *.o $(TARGET); rm -f ./lib/*.so; \
	make -C DBoW2 clean

install: $(TARGET)
	make -C DBoW2 install && cp DBoW2/libDBoW2.so ./lib/

uninstall:
	make -C DBoW2 uninstall
