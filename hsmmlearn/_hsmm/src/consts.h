// global constants
#include <string>

#ifndef CONSTS
#define CONSTS

// -- the mode in which the program is running: release, storing the parameters of the call of the FB resp. Viterbi,
// -- testing the FB resp. Viterbi using the stored parameters
enum Run_Mode { RELEASE, STORE_FB, STORE_VIT, TEST_FB, TEST_VIT };

const Run_Mode run_mode = RELEASE;
const std::string PARA_FNAME = "C:\\svn\\hsmm\\para.txt";
const int noCensoring = 0, rightCensoring = 1, leftRightCensoring = 2;

#endif

