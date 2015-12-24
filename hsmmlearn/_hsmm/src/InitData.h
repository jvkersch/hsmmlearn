#ifndef INITDATA
	#define INITDATA

	#include "matrix.h"
	#include "cube.h"
	#include <stdio.h>
	#include <vector>

	using namespace std;
	
	extern double** StateIn;
	extern double** F;
	extern double** L; 
	extern double** G; 
	extern double*** H;
	extern double** L1;  
	extern double* N;  
	extern double** Norm;  
	extern double** d; 
	extern double** D;  
	extern double* mean_d;
	extern double** p;
	extern double* pi;  																																																																																						 																																																																																						
	extern double** eta;  																																																																																	 
	extern double** xi;  																																																																																	 
	extern double** alpha;
	extern int** maxI;
	extern int** maxU;
	extern double** pdf;
	extern int* hiddenStates;

	extern int J, Y, tau, M;
	extern int Censoring, Output;
	extern bool LeftCensoring, RightCensoring;

	int InitInputData(char inputFilename[], double InputData[]);
	void InitOutputData(char outputPath[]);
	void InitParaAndVar(int CensoringPara, int tauPara, int JPara, 
						int MPara, double dPara[], double pPara[], double piPara[], double pdfPara[]);
	void freeMemory();

#endif
