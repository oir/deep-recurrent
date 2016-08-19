# compile & run
default:
	g++ drnt.cpp -I ./Eigen/ -std=c++11 -O3 -o drnt
	g++ evaluator.cpp -I ./Eigen/ -std=c++11 -O3 -o evaluator

clean:
	rm -f drnt evaluator
