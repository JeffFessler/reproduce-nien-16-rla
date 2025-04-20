#include "mex.h"
#include <string.h>

/***** input *****/
#define _i_A    prhs[0]

/***** output *****/
#define _o_pr   plhs[0]
#define _o_ir   plhs[1]
#define _o_jc   plhs[2]

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mwIndex *ir, *jc;
    double *pr;
    mwSize nzmax, m, n;

    ir    = mxGetIr(_i_A);
    jc    = mxGetJc(_i_A);
    pr    = mxGetPr(_i_A);
    nzmax = mxGetNzmax(_i_A);
    n     = mxGetN(_i_A);
    
    _o_pr = mxCreateNumericMatrix(nzmax, 1, mxDOUBLE_CLASS, mxREAL);
    memcpy((void*)mxGetPr(_o_pr), (const void*)pr, nzmax*sizeof(double));
    
    _o_ir = mxCreateNumericMatrix(nzmax, 1, mxUINT64_CLASS, mxREAL);
    memcpy((void*)mxGetPr(_o_ir), (const void*)ir, nzmax*sizeof(unsigned long));
    
    _o_jc = mxCreateNumericMatrix(n+1, 1, mxUINT64_CLASS, mxREAL);
    memcpy((void*)mxGetPr(_o_jc), (const void*)jc, (n+1)*sizeof(unsigned long));
}
