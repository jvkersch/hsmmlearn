#include "ViterbiImpl.h"
#include "InitData.h"
#include "dCalc.h"
#include "error.h"
#include "consts.h"
#include <time.h>
#include <math.h>
#include <fstream>
#include <iostream>



void ViterbiImpl(int tauPara, int JPara, int MPara,
                 double dPara[], double pPara[], double piPara[], double pdfPara[], int hiddenStatesPara[])
{
    int i, j, k = 0, k_alt, t, u, dummyInt = 0;
    double Observ, x = 0;
    bool first, first_i, first_alpha;

    // output all function parameters to file
    if (run_mode == STORE_VIT) {
        ofstream ofs(PARA_FNAME.c_str());
        if (!ofs) {
            cerr << "unable to open file: " << PARA_FNAME << endl;
            exit(0);
        }

        // output tau, J, and M
        ofs << 0 << endl;   // dummy output for CensoringPara
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

    InitParaAndVar(dummyInt, tauPara, JPara, MPara, dPara, pPara, piPara, pdfPara);

    CalcStoreD();

    for (t = 0; t <= tau - 1; t++) {
        for (j = 0; j <= J - 1; j++) {
            Observ = 0;
            first_alpha = true;
            for (u = 1; u <= min(t, M); u++) {
                first_i = true;
                for (i = 0; i <= J - 1; i++)
                    if (i != j)
                        if ((log(p[i][j]) + alpha[i][t - u] > x) || first_i) {
                            x = log(p[i][j]) + alpha[i][t - u];
                            k = i;
                            first_i = false;
                        }
                if (first_alpha || (Observ + log(d[j][u]) + x > alpha[j][t])) {
                    alpha[j][t] = Observ + log(d[j][u]) + x;
                    maxU[j][t] = u;
                    maxI[j][t] = k;
                    first_alpha = false;
                }
                Observ += log(pdf[j][t - u]);
            }
            if (t + 1 <= M) {
                if (first_alpha || (Observ + log(d[j][t + 1] * pi[j]) > alpha[j][t])) {
                    alpha[j][t] = Observ + log(d[j][t + 1] * pi[j]);
                    maxU[j][t] = -1;
                    maxI[j][t] = -1;
                }
            }
            alpha[j][t] += log(pdf[j][t]);
        }
    }

    for (j = 0; j <= J - 1; j++)
    {
        Observ = 0;
        first_alpha = true;
        for (u = 1; u <= tau - 1; u++)
        {
            first_i = true;
            for (i = 0; i <= J - 1; i++)
                if (i != j)
                    if ((log(p[i][j]) + alpha[i][tau - 1 - u] > x) || first_i)
                    {
                        x = log(p[i][j]) + alpha[i][tau - 1 - u];
                        k = i;
                        first_i = false;
                    }
            if ((Observ + log(D[j][u]) + x > alpha[j][tau - 1]) || first_alpha)
            {
                alpha[j][tau - 1] = Observ + log(D[j][u]) + x;
                maxU[j][tau - 1] = u;
                maxI[j][tau - 1] = k;
                first_alpha = false;
            }
            Observ += log(pdf[j][tau - 1 - u]);
        }
        if ((Observ + log(D[j][tau - 1] * pi[j]) > alpha[j][tau - 1]) || first_alpha)
        {
            alpha[j][tau - 1] = Observ + log(D[j][tau] * pi[j]);
            maxU[j][tau - 1] = -1;
            maxI[j][tau - 1] = -1;
        }
        alpha[j][tau - 1] += log(pdf[j][tau - 1]);
    }

    // Save result
    first = true;
    for (j = 0; j <= J - 1; j++)
    {
        if ((alpha[j][tau - 1] > x) || first)
        {
            x = alpha[j][tau - 1];
            k = j;
            first = false;
        }
    }

    t = tau - 1;
    while (maxI[k][t] >= 0)
    {
        for (i = t; i >= t - maxU[k][t] + 1; i--)
        {
            hiddenStates[i] = k;
        }
        k_alt = k;
        k = maxI[k][t];
        t -= maxU[k_alt][t];
    }

    for (i = t; i >= 0; i--)
        hiddenStates[i] = k;

    // Save parameters
    for (t = 0; t < tau; t++)
    {
        hiddenStatesPara[t] = hiddenStates[t];
    }

    freeMemory();
}
