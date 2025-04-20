#include "mex.h"

/***** input *****/
#define _i_A    prhs[0]
#define _i_y    prhs[1]

/***** output *****/
#define _o_z    plhs[0]

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* unsigned int (uint64) is enough */
    unsigned int *ir, *jc;
    float *pr, *y, *z;
    unsigned int m, n;
    mxArray *tmp;
    
    tmp = mxGetField(_i_A, 0, "pr");
    pr = (float *)mxGetPr(tmp);
    
    tmp = mxGetField(_i_A, 0, "ir");
    ir = (unsigned int*)mxGetPr(tmp);
    
    tmp = mxGetField(_i_A, 0, "jc");
    jc = (unsigned int*)mxGetPr(tmp);
    
    tmp = mxGetField(_i_A, 0, "m");
    m = (unsigned int)mxGetScalar(tmp);    
    
    tmp = mxGetField(_i_A, 0, "n");
    n = (unsigned int)mxGetScalar(tmp);

    y = (float *)mxGetPr(_i_y);

    _o_z = mxCreateNumericMatrix(n, 1, mxSINGLE_CLASS, mxREAL);
    z = (float *)mxGetPr(_o_z);

    for (int j=0; j<n; j++) {
        for (int k=jc[j]; k<jc[j+1]; k++) {
            z[j] += pr[k]*y[ir[k]];
        }
    }
}
