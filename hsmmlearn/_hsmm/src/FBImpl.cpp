#include "FBImpl.h"
#include "dCalc.h"
#include "Mathe.h"
#include "InitData.h"
#include "error.h"
#include "matrix.h"
#include "cube.h"
#include "consts.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <limits.h>
#include <iostream>
#include <fstream>

using namespace std;



double** StateIn = NULL;
double** F = NULL;
double** L = NULL;
double** G = NULL;
double*** H = NULL;
double** L1 = NULL;
double* N = NULL;
double** Norm = NULL;
double** d = NULL;
double** D = NULL;
double* mean_d = NULL;
double** p = NULL;
double* pi = NULL;
double** eta = NULL;
double** xi = NULL;
double** alpha = NULL;
int** maxI = NULL;
int** maxU = NULL;
double** pdf = NULL;
int* hiddenStates = NULL;



int J, tau, M;
int Censoring, Output;
bool LeftCensoring, RightCensoring;

void FBImpl(int CensoringPara, int tauPara, int JPara, int MPara,
            double dPara[], double pPara[], double piPara[], double pdfPara[],
            double FPara[], double LPara[], double GPara[], double L1Para[],
            double NPara[], double NormPara[],
            double etaPara[], double xiPara[], int *err)
{
    int i, j, k, t, u, v;
    double Observ, r, s, w;

    // output all function parameters to file
    if (run_mode == STORE_FB) {
        ofstream ofs(PARA_FNAME.c_str());
        if (!ofs) {
            cerr << "unable to open file: " << PARA_FNAME << endl;
            exit(0);
        }

        // output tau, J, and M
        ofs << CensoringPara << endl;
        ofs << tauPara << endl;
        ofs << JPara << endl;
        ofs << MPara << endl << endl;

        // output d
        for (int j = 0; j < JPara; ++j) {
            for (int m = 0; m < MPara; ++m) {
                ofs << j << " " << m << " " << dPara[j * MPara + m] << endl;
            }
        }
        ofs << endl;

        // output p
        for (int j = 0; j < JPara; ++j) {
            for (int k = 0; k < JPara; ++k) {
                ofs << j << " " << k << " " << pPara[j * JPara + k] << endl;
            }
        }
        ofs << endl;

        // output pi
        for (int j = 0; j < JPara; ++j) {
            ofs << j << " " << piPara[j] << endl;
        }
        ofs << endl;

        // output pdf
        for (int j = 0; j < JPara; ++j) {
            for (int t = 0; t < tauPara; ++t) {
                ofs << j << " " << t << " " << pdfPara[j * tauPara + t] << endl;
            }
        }
        ofs << endl;
    }

    try {
        InitParaAndVar(CensoringPara, tauPara, JPara, MPara, dPara, pPara, piPara, pdfPara);

        CalcStoreD();

        // forward recursion
        for (t = 0; t <= tau - 1; t++)
        {
            N[t] = 0;
            for (j = 0; j <= J - 1; j++)
            {
                if (t == 0)
                {
                    Norm[j][0] = pdf[j][0] * pi[j];
                }
                else
                    Norm[j][t] = pdf[j][t] * (StateIn[j][t] - F[j][t - 1] + Norm[j][t - 1]);
                N[t] += Norm[j][t];
            }

            if (N[t] <= 0)
            {
                throw var_nonpositive_exception();
            }

            for (j = 0; j <= J - 1; j++)
            {
                Norm[j][t] /= N[t];
            }

            for (j = 0; j <= J - 1; j++)
            {
                F[j][t] = 0;
                Observ = 1;

                if (t < tau - 1)
                {
                    for (u = 1; u <= min(t + 1, M); u++)
                    {
                        Observ *= pdf[j][t - u + 1] / N[t - u + 1];

                        if (u < t + 1)
                        {
                            F[j][t] += Observ * d[j][u] * StateIn[j][t - u + 1];
                        }
                        else
                        {
                            if (LeftCensoring)
                                F[j][t] += Observ * D[j][t + 1] / mean_d[j] * pi[j];
                            else
                                F[j][t] += Observ * d[j][t + 1] * pi[j];
                        }
                    }
                }
                else
                {
                    for (u = 1; u <= min(tau, M); u++)
                    {
                        Observ *= pdf[j][tau - u] / N[tau - u];
                        if (u < tau)
                        {
                            F[j][tau - 1] += Observ * D[j][u] * StateIn[j][tau - u];
                        }
                        else
                        {
                            if (LeftCensoring)
                            {
                                w = 0;
                                for (v = tau; v <= M; v++)
                                    w += D[j][v];
                                F[j][tau - 1] += Observ * w / mean_d[j] * pi[j];
                            }
                            else
                                F[j][tau - 1] += Observ * D[j][tau] * pi[j];
                        }
                    }
                }

                if (F[j][t] <= 0)
                {
                    throw var_nonpositive_exception();
                }
            }

            if (t < tau - 1)
            {
                for (j = 0; j <= J - 1; j++)
                {
                    StateIn[j][t + 1] = 0;
                    for (i = 0; i <= J - 1; i++)
                    {
                        StateIn[j][t + 1] += p[i][j] * F[i][t];
                    }
                }
            }
        }

        // Backward recursion
        for (j = 0; j <= J - 1; j++)
            L[j][tau - 1] = F[j][tau - 1];

        for (t = tau - 2; t >= 0; t--)
        {
            for (j = 0; j <= J - 1; j++)
            {
                G[j][t + 1] = 0;

                Observ = 1;
                for (u = 1; u <= min(tau - 1 - t, M); u++)
                {
                    Observ *= pdf[j][t + u] / N[t + u];

                    if (u < tau - 1 - t)
                    {
                        H[j][t + 1][u] = L1[j][t + u] * Observ * d[j][u] / F[j][t + u];
                    }
                    else
                    {
                        H[j][t + 1][u] = Observ * D[j][tau - 1 - t];
                    }
                    G[j][t + 1] += H[j][t + 1][u];
                }
            }

            for (j = 0; j <= J - 1; j++)
            {
                L1[j][t] = 0;
                for (k = 0; k <= J - 1; k++)
                    L1[j][t] += G[k][t + 1] * p[j][k];
                L1[j][t] *= F[j][t];
                L[j][t] = L1[j][t] + L[j][t + 1] - G[j][t + 1] * StateIn[j][t + 1];
            }
        }

        // Calculation of eta and xi
        if (LeftCensoring)
        {
            // Calculation eta
            for (j = 0; j <= J - 1; j++)
            {
                for (u = 1; u <= M; u++)
                {
                    r = 1;
                    w = 0;
                    for (t = 1; t <= min(u, tau - 1); t++)
                    {
                        r = 1;
                        for (v = 1; v <= t; v++)
                            r *= pdf[j][t - v] / N[t - v];
                        w += L1[j][t - 1] / F[j][t - 1] * r;
                    }
                    eta[j][u] = w * d[j][u] / mean_d[j] * pi[j];
                    if (u >= tau)
                    {
                        r = 1;
                        for (v = 1; v <= tau; v++)
                            r *= pdf[j][tau - v] / N[tau - v];
                        eta[j][u] += r * (u + 1 - tau) * d[j][u] / mean_d[j] * pi[j];
                    }
                }
            }
            // Calculation xi
            for (j = 0; j <= J - 1; j++)
            {
                for (u = 1; u <= M; u++)
                {
                    w = 0;
                    for (t = 0; t <= tau - 2; t++)
                    {
                        r = 0;
                        for (i = 0; i <= J - 1; i++)
                            if (i != j)  r += p[i][j] * F[i][t];
                        if (u <= tau - 2 - t )
                            r *= H[j][t + 1][u];
                        else
                        {
                            r *= d[j][u];
                            for (v = 0; v <= tau - 2 - t; v++)
                                r *= pdf[j][tau - 1 - v] / N[tau - 1 - v];
                        }
                        w += r;
                    }
                    xi[j][u] = w;
                }
            }
        }
        else
        {
            // Calculation eta
            for (j = 0; j <= J - 1; j++)
            {
                for (u = 1; u <= M; u++)
                {
                    w = 0;
                    s = 1;
                    for (t = tau - 2 - ind(!(RightCensoring)) * u; t >= 0; t--)
                    {
                        r = 0;
                        for (i = 0; i <= J - 1; i++)
                            if (i != j)  r += p[i][j] * F[i][t];
                        if (u <= tau - 2 - t )
                        {
                            r *= H[j][t + 1][u];
                        }
                        else
                        {
                            s *= pdf[j][t + 1] / N[t + 1];
                            r *= s * d[j][u];
                        }
                        w += r;
                    }
                    if ((RightCensoring) || (u <= tau - 1))
                    {
                        r = d[j][u] * pi[j];
                        if (u <= tau - 1)
                        {
                            for (v = 1; v <= u; v++)
                                r *= pdf[j][u - v] / N[u - v];
                            r *= L1[j][u - 1] / F[j][u - 1];
                        }
                        else
                            for (v = 1; v <= tau; v++)
                                r *= pdf[j][tau - v] / N[tau - v];
                        w += r;
                    }
                    eta[j][u] = w;
                }
            }
        }

        // Save parameters
        for (j = 0; j <= J - 1; j++) {
            for (t = 0; t < tau; t++) {
                FPara[j * tau + t] = F[j][t];
                LPara[j * tau + t] = L[j][t];
                NormPara[j * tau + t] = Norm[j][t];
            }
            for (t = 1; t < tau; t++)
                GPara[j * tau + t] = G[j][t];
            for (t = 0; t < tau - 1; t++)
                L1Para[j * tau + t] = L1[j][t];
            for (t = 0; t < M; t++)
                etaPara[j * M + t] = eta[j][t + 1];
            if (LeftCensoring)
                for (t = 0; t < M; t++)
                    xiPara[j * M + t] = xi[j][t + 1];
        }
        for (t = 0; t < tau; t++) {
            NPara[t] = N[t];
        }
    }
    catch (var_nonpositive_exception e)
    {
        *err = 1;
    }
    catch (memory_exception e)
    {
        *err = 2;
    }
    catch (file_exception e)
    {
        *err = 3;
    }

    freeMemory();
}
