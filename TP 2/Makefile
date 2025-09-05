CC = g++
CXXFLAGS = -O3 -I ./Eigen

TARGET = metodoPotencia
SOURCES = metodoPotencia.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET) unzip

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) *.txt

unzip:
	unzip autovectores_valores.zip