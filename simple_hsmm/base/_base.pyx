#cython: boundscheck=False

cdef extern from "../_hsmm/src/ViterbiImpl.h":
    void ViterbiImpl(int tauPara, int JPara, int MPara,
                     double dPara[], double pPara[], double piPara[],
                     double pdfPara[], int hiddenStatesPara[]) nogil

cdef extern from "../_hsmm/src/FBImpl.h":
    void FBImpl(int CensoringPara, int tauPara, int JPara,
                int MPara, double dPara[], double pPara[], double piPara[],
                double pdfPara[],
                double FPara[], double LPara[], double GPara[], double L1Para[],
                double NPara[], double NormPara[], 
                double etaPara[], double xiPara[], int *err) nogil;


cpdef _viterbi_impl(int tau_para, int j_para, int m_para,
                    double[:] d_para, double[:] p_para, double[:] pi_para,
                    double[:] pdf_para, int[:] hidden_states_para):
    ViterbiImpl(tau_para, j_para, m_para, &d_para[0], &p_para[0], &pi_para[0],
                &pdf_para[0], &hidden_states_para[0])


cpdef int _fb_impl(int censoring_para, int tau_para, int j_para, int m_para,
               double[:] d_para, double[:] p_para, double[:] pi_para,
               double[:] pdf_para, double[:] f_para, double[:] l_para,
               double[:] g_para, double[:] l1_para, double[:] n_para,
               double[:] norm_para, double[:] eta_para, double[:] xi_para):

     cdef int err = 0

     FBImpl(censoring_para, tau_para, j_para, m_para,
            &d_para[0], &p_para[0], &pi_para[0],
            &pdf_para[0], &f_para[0], &l_para[0],
            &g_para[0], &l1_para[0], &n_para[0],
            &norm_para[0], &eta_para[0], &xi_para[0],
            &err)

     return err
