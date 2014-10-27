CC=gcc
CFLAGS=-DNDEBUG -fPIC -O3 -Wall \
  $(shell pkg-config --cflags opencv)
LFLAGS=-shared $(shell pkg-config --libs opencv)

LIBNAME=DBoW2
PREFIX=/usr/local

INSTALL_LIB=$(PREFIX)/lib
INSTALL_INCLUDE=$(PREFIX)/include

TARGET=lib$(LIBNAME).so

SRCS=$(wildcard *.cpp)

all: $(TARGET)

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CC) $(CFLAGS) -MM $^ > ./.depend;

ifneq ($(MAKECMDGOALS),clean)
	-include .depend
endif

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ 

$(TARGET): $(SRCS:.cpp=.o)
	$(CC) $^ $(LFLAGS) -o $@

clean:
	rm -f .depend *.o $(TARGET)

install: $(TARGET)
	cp $(TARGET) $(INSTALL_LIB)/ && \
	mkdir -p $(INSTALL_INCLUDE)/$(LIBNAME) && \
	cp *.h $(INSTALL_INCLUDE)/$(LIBNAME)/

uninstall: 
	rm $(INSTALL_LIB)/$(TARGET) && \
	rm $(INSTALL_INCLUDE)/$(LIBNAME)/* && \
	rmdir $(INSTALL_INCLUDE)/$(LIBNAME)

