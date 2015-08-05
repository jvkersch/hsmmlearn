#include <R_ext/libextern.h>
#ifndef RINTERFACE
#define RINTERFACE

extern "C" {
	void LibExport FB(int *CensoringPara, int *tauPara, int *JPara, int *MPara, 
					  double *dPara, double *pPara, double *piPara, double *pdfPara,
					  double *F, double *L, double *G, double *L1,
					  double *N, double *Norm, double *eta, double *xi, int *err);
	}
	
extern "C" {
	void LibExport
		Viterbi(int *tauPara, int *JPara, int *MPara, 
				double *dPara, double *pPara, double *piPara, double *pdfPara, int *hiddenStates);
}
#endif

