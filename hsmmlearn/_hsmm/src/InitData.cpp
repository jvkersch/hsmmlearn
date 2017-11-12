#include "consts.h"
#include "InitData.h"
#include "matrix.h"
#include "cube.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

using namespace std;



int InitInputData(char inputFilename[], double InputData[])
{
    int i;
    char str[100];
    FILE *stream;

    stream = fopen(inputFilename, "r" );
    i = 0;
    while (fgets(str, 100, stream) != NULL)
    {
        InputData[i] = atof(str);
        i++;
    }
    fclose(stream);

    return i;
}


void InitParaAndVar(int CensoringPara,
                    int tauPara, int JPara, int MPara, double dPara[],
                    double pPara[], double piPara[], double pdfPara[])
{
    int i, j, u;

    Censoring = CensoringPara;

    tau = tauPara;
    J = JPara;
    M = MPara;

    StateIn = matrix<double>(J, tau);
    if (StateIn == NULL)
    {
        throw memory_exception();
    }

    F = matrix<double>(J, tau);
    if (F == NULL)
    {
        throw memory_exception();
    }
    
    L = matrix<double>(J, tau);
    if (L == NULL)
    {
        throw memory_exception();
    }

    G = matrix<double>(J, tau);
    if (G == NULL)
    {
        throw memory_exception();
    }

    H = cube<double>(J, tau, M + 1);
    if (H == NULL)
    {
        throw memory_exception();
    }

    L1 = matrix<double>(J, tau);
    if (L1 == NULL)
    {
        throw memory_exception();
    }

    N = new double[tau];
    if (N == NULL)
    {
        throw memory_exception();
    }

    Norm = matrix<double>(J, tau);
    if (Norm == NULL)
    {
        throw memory_exception();
    }

    d = matrix<double>(J, M + 1);
    if (d == NULL)
    {
        throw memory_exception();
    }

    if (M + 1 > tau)
        D = matrix<double>(J, M + 1);
    else
        D = matrix<double>(J, tau + 1);
    if (D == NULL)
    {
        throw memory_exception();
    }

    mean_d = new double[J];
    if (mean_d == NULL)
    {
        throw memory_exception();
    }

    p = matrix<double>(J, J);
    if (p == NULL)
    {
        throw memory_exception();
    }

    pi = new double[J];
    if (pi == NULL)
    {
        throw memory_exception();
    }

    eta = matrix<double>(J, M + 1);
    if (eta == NULL)
    {
        throw memory_exception();
    }

    xi = matrix<double>(J, M + 1);
    if (xi == NULL)
    {
        throw memory_exception();
    }

    alpha = matrix<double>(J, tau);
    if (alpha == NULL)
    {
        throw memory_exception();
    }

    maxI = matrix<int>(J, tau);
    if (maxI == NULL)
    {
        throw memory_exception();
    }

    maxU = matrix<int>(J, tau);
    if (maxU == NULL)
    {
        throw memory_exception();
    }

    pdf = matrix<double>(J, tau);
    if (pdf == NULL)
    {
        throw memory_exception();
    }

    hiddenStates = new int[tau];
    if (hiddenStates == NULL)
    {
        throw memory_exception();
    }


    for (j = 0; j <= J - 1; j++)
    {
        pi[j] = piPara[j];
    }

    for (j = 0; j <= J - 1; j++)
        for (u = 0; u <= M - 1; u++) {
            d[j][u + 1] = dPara[j * M + u];
        }

    for (i = 0; i <= J - 1; i++)
        for (j = 0; j <= J - 1; j++)
        {
            p[i][j] = pPara[i * J + j];
        }

    for (j = 0; j <= J - 1; j++)
        for (u = 0; u <= tau - 1; u++)
        {
            pdf[j][u] = pdfPara[j * tau + u];
        }

    switch (Censoring)
    {
    case noCensoring:
    {
        LeftCensoring = false;
        RightCensoring = false;
        break;
    }
    case rightCensoring:
    {
        LeftCensoring = false;
        RightCensoring = true;
        break;
    }
    case leftRightCensoring:
    {
        LeftCensoring = true;
        RightCensoring = true;
        break;
    }
    }
}


void freeMemory()
{
    if (StateIn != NULL) free_matrix(StateIn);
    if (L != NULL) free_matrix(L);
    if (G != NULL) free_matrix(G);
    if (H != NULL) free_cube(H);
    if (L1 != NULL) free_matrix(L1);
    if (N != NULL) delete [] N;
    if (Norm != NULL) free_matrix(Norm);
    if (d != NULL) free_matrix(d);
    if (D != NULL) free_matrix(D);
    if (mean_d != NULL) delete [] mean_d;
    if (p != NULL) free_matrix(p);
    if (pi != NULL) delete [] pi;
    if (eta != NULL) free_matrix(eta);
    if (xi != NULL) free_matrix(xi);
    if (alpha != NULL) free_matrix(alpha);
    if (maxI != NULL) free_matrix(maxI);
    if (maxU != NULL) free_matrix(maxU);
    if (pdf != NULL) free_matrix(pdf);
}
