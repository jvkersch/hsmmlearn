#ifndef DCALC
	#define DCALC

	#include "matrix.h"
	#include "cube.h"
	#include <stdio.h>
	#include <vector>

	using namespace std;
	
	#if DEBUG_ARRAY_STATIC
		extern double D[MaxJ][MaxTau + 1]; 
		extern double d[MaxJ][MaxRunlength + 1]; 
		extern double mean_d[MaxJ];
	#else
		extern double** D; 
		extern double** d; 	
		extern double* mean_d;
	#endif

	extern int J, tau, M;

	void CalcStoreD();
	
#endif
