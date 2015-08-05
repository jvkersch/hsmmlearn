#ifndef VITERBIIMPL
#define VITERBIIMPL
#include "matrix.h"
#include "cube.h"
#include <stdio.h>
#include <time.h>

#if DEBUG_ARRAY_STATIC
	double d[MaxJ][MaxRunlength + 1];
	double D[MaxJ][MaxTau + 1];
	double p[MaxJ][MaxJ]; 
	double pi[MaxJ];  																																																																																							
	double alpha[MaxJ][MaxTau];
	int maxI[MaxJ][MaxTau];
	int maxU[MaxJ][MaxTau];
	double pdf[MaxJ][MaxTau];
	int hiddenStates[MaxTau];
#else
	extern double** d;
	extern double** D;
	extern double** p;
	extern double* pi;  																																																																																						 																																																																																						
	extern double** alpha;
	extern int** maxI;
	extern int** maxU;
	extern double** pdf;
	extern int* hiddenStates;
#endif

extern int J, tau, M;
extern int Output;

void ViterbiImpl(int tauPara, int JPara, int MPara, 
				 double dPara[], double pPara[], double piPara[], double pdfPara[], int hiddenStatesPara[]);
#endif
