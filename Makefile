CC=gcc
CFLAGS=-DNDEBUG -O3 -Wall -IDUtils -IDUtilsCV -IDVision -IDBoW2 \
  $(shell pkg-config --cflags opencv)
LFLAGS=lib/libDBoW2.so lib/libDVision.so lib/libDUtilsCV.so lib/libDUtils.so \
  $(shell pkg-config --libs opencv) -lstdc++

DEPS=lib/libDUtils.so lib/libDUtilsCV.so lib/libDVision.so lib/libDBoW2.so
TARGET=demo

all: $(TARGET) 

$(TARGET): $(TARGET).o $(DEPS)
	$(CC) $(TARGET).o $(LFLAGS) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ 

lib/libDUtils.so:
	make -C DUtils && mkdir -p ./lib/ && cp DUtils/libDUtils.so ./lib/

lib/libDUtilsCV.so:
	make -C DUtilsCV && mkdir -p ./lib/ && cp DUtilsCV/libDUtilsCV.so ./lib/
	
lib/libDVision.so:
	make -C DVision && mkdir -p ./lib/ && cp DVision/libDVision.so ./lib/

lib/libDBoW2.so:
	make -C DBoW2 && mkdir -p ./lib/ && cp DBoW2/libDBoW2.so ./lib/

clean:
	rm -f *.o $(TARGET); rm -f ./lib/*.so; \
	make -C DUtils clean; \
	make -C DUtilsCV clean; \
	make -C DVision clean; \
	make -C DBoW2 clean

install: $(TARGET)
	make -C DUtils install && cp DUtils/libDUtils.so ./lib/ && \
	make -C DUtilsCV install && cp DUtilsCV/libDUtilsCV.so ./lib/ && \
	make -C DVision install && cp DVision/libDVision.so ./lib/ && \
	make -C DBoW2 install && cp DBoW2/libDBoW2.so ./lib/

uninstall:
	make -C DUtils uninstall; \
	make -C DUtilsCV uninstall; \
	make -C DVision uninstall; \
	make -C DBoW2 uninstall

