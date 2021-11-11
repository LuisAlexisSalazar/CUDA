
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <random>
#include <string>
using namespace std;


float LIMIT_L = 1.f;
float LIMIT_R = 100.f;
random_device rd;
mt19937 mt(rd());
uniform_real_distribution<float> dist(LIMIT_L, LIMIT_R);

//Ctrl+ shift + space
//ctrl + k + ctrl + d


__global__
void vecAddkernel(float* A, float* B, float* C, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
		C[i] = A[i] + B[i];
}



void vecAdd(float* A, float* B, float* C, int n) {
	int size = n * sizeof(float);
	float* d_A, * d_B, * d_C;
	cudaMalloc((void**)&d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_C, size);
	
	//Función kernel
	vecAddkernel<<<ceil(n/256.0),256>>>(d_A, d_B, d_C,n);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	//Print Vector Resultante

	//Liberara memoria del device A,B,C
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

}

void fillVector(float* Vect, int n) {
	for (int i = 0; i < n; i++)
		Vect[i] = dist(mt);
}

void printVector(float* Vect, int n, string nameVect = "") {
	cout << endl;
	if (nameVect != "")
		cout << nameVect << " : ";

	for (int i = 0; i < n; i++)
		cout << Vect[i] << "	";
}


int main()
{
	int n = 5;
	float* A = new float[n];
	float* B = new float[n];
	float* C = new float[n];

	fillVector(A, n);
	fillVector(B, n);

	printVector(A, n,"A");
	printVector(B, n,"B");
		
	vecAdd(A,B,C,n);
	printVector(C, n, "C");


	return 0;
}

