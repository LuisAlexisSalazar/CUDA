#include <cuda_runtime.h>

// ctrl+shift+space to see parameters 
//Formatear code ctr+k ctrl+d
//Scroll barra es información general
int main() {
	int* a;
	cudaMalloc(&a, 100);

	cudaFree(a);

	return 0;

}