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
    """
    Low-level Viterbi wrapper.

    Parameters
    ----------
    tau : int
        Number of observations.
    j : int
        Number of internal states.
    m : int
        Number of durations. Permissible durations are 1 through m (inclusive).
    d : array, shape=(j * m, )
        Probabilities for the individual durations, where the first m elements
        specify the duration distribution for state 0, and so on.
    p : array, shape=(j * j, )
        Transition matrix in row-major order.
    pi : array, shape=(j, )
        Initial state probabilities.
    pdf : array, shape(j * tau, )
        Likelihoods of the observations given the states. The first tau
        elements are the likelihoods in state 0, and so on.
    states : array, shape(tau, )
        Output array for the reconstructed internal states.

    """

    ViterbiImpl(tau_para, j_para, m_para, &d_para[0], &p_para[0], &pi_para[0],
                &pdf_para[0], &hidden_states_para[0])


cpdef int _fb_impl(int censoring_para, int tau_para, int j_para, int m_para,
               double[:] d_para, double[:] p_para, double[:] pi_para,
               double[:] pdf_para, double[:] f_para, double[:] l_para,
               double[:] g_para, double[:] l1_para, double[:] n_para,
               double[:] norm_para, double[:] eta_para, double[:] xi_para):
    """ Low level EM wrapper.

    Does one iteration of the forward-backward algorithm.

    Parameters
    ----------
    censoring_para : int
        If equal to 1, the last visited state contributes to the likelihood.
        If equal to 0, the partial likelihood estimator, which ignores the
        contribution of the last visited state, is used.
    tau_para : int
        Number of observations.
    j_para : int
        Number of internal states.
    m_para : int
        Number of durations. Permissible durations are 1 through m (inclusive)
    d_para : array, shape=(j*m, )
        Probabilities for the individual durations, where the first m elements
        specify the duration distribution for state 0, and so on.
    p_para : array, shape=(j * j, )
        Transition matrix in row-major order.
    pi_para : array, shape=(j, )
        Initial state probabilities.
    pdf_para : array, shape(j * tau, )
        Likelihoods of the observations given the states. The first tau
        elements are the likelihoods in state 0, and so on.
    f_para : array, shape(j * tau, )
    l_para : array, shape(j * tau, )
    g_para : array, shape(j * tau, )
    l1_para : array, shape(j * tau, )
    n_para : array, shape(tau, )
    norm_para : array, shape(j * tau, )
    eta_para : array, shape(j * m, )
    xi_para : array, shape(j * m, )

    """

    cdef int err = 0

    FBImpl(censoring_para, tau_para, j_para, m_para,
           &d_para[0], &p_para[0], &pi_para[0],
           &pdf_para[0], &f_para[0], &l_para[0],
           &g_para[0], &l1_para[0], &n_para[0],
           &norm_para[0], &eta_para[0], &xi_para[0],
           &err)

    return err
