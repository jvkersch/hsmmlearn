#include "RInterface.h"
#include "FBImpl.h"
#include "ViterbiImpl.h"

void FB(int *CensoringPara, int	*tauPara, int *JPara, int *MPara, 
		double *dPara, double *pPara, double *piPara, double *pdfPara,
		double *FPara, double *LPara, double *GPara, double *L1Para, double *NPara, double *NormPara, 		
		double *etaPara, double *xiPara, int *err) 
{
	FBImpl(*CensoringPara, *tauPara, *JPara, *MPara, dPara, pPara, piPara, pdfPara, 
		   FPara, LPara, GPara, L1Para, NPara, NormPara, etaPara, xiPara, err);	
}



void Viterbi(int *tauPara, int *JPara, int *MPara, 
			 double *dPara, double *pPara, double *piPara, double *pdfPara, int *hiddenStatesPara)
{
	ViterbiImpl(*tauPara, *JPara, *MPara, dPara, pPara, piPara, pdfPara, hiddenStatesPara);	
}
