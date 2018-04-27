proj3: mpi.o bitmap.o pthread.o openmp.o
	mpic++ -o mpi mpi.o bitmap.o 
	g++ -o pthread pthread.o bitmap.o -lpthread
	g++ -o openmp openmp.o bitmap.o -fopenmp
mpi.o: mpi.cpp
	mpic++ -std=c++11 -c mpi.cpp
pthread.o: pthread.cpp
	g++ -std=c++11 -c pthread.cpp
openmp.o: openmp.cpp
	g++ -std=c++11 -c openmp.cpp -fopenmp
bitmap.o: bitmap.cpp
	g++ -std=c++11 -c bitmap.cpp
clean:
	rm mpi openmp pthread mpi.o bitmap.o pthread.o openmp.o 
