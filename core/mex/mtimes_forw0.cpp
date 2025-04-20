#include "mex.h"

/***** input *****/
#define _i_A    prhs[0]
#define _i_x    prhs[1]

/***** output *****/
#define _o_y    plhs[0]

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mwIndex *ir, *jc;
    float *pr, *x, *y;
    mwSize nd, np;

    ir = mxGetIr(_i_A);
    jc = mxGetJc(_i_A);
    pr = (float *)mxGetPr(_i_A);
    nd = mxGetM(_i_A);
    np = mxGetN(_i_A);

    x  = (float *)mxGetPr(_i_x);

    mwSize odim[] = {nd, 1};
    _o_y = mxCreateNumericArray(2, odim, mxSINGLE_CLASS, mxREAL);
    y  = (float *)mxGetPr(_o_y);

    for (int j=0; j<np; j++) {
        for (int k=jc[j]; k<jc[j+1]; k++) {
            y[ir[k]] += pr[k]*x[j];
        }
    }
}
