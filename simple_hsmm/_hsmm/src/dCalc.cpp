#include "dCalc.h"
#include <iostream>

using namespace std;



void CalcStoreD()
{
	double x;
	int j, u, v;

	// Store D
	for (j = 0; j < J; j++) {
		for (u = 1; u <= M; u++) {
			x = 0;
			for (v = u; v < M + 1; v++)
				x += d[j][v];
			D[j][u] = x;			
		}
		for (u = M + 1; u <= tau; u++)
		{
			D[j][u] = 0;
		}
	}

	// Store mean_d
	for (j = 0; j < J; j++) {
		x = 0;
		for (v = 1; v < M + 1; v++) {
			x += d[j][v] * v;
		}
		mean_d[j] = x;
	}
}


