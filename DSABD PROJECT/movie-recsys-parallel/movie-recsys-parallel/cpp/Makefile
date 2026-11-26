CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra

recsys: recsys.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f recsys
