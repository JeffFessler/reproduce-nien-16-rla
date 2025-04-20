#include "mex.h"
#include <string.h>

/***** input *****/
#define _i_pr      prhs[0]
#define _i_ir      prhs[1]
#define _i_jc      prhs[2]
#define _i_nzmax   prhs[3]
#define _i_m       prhs[4]
#define _i_n       prhs[5]

/***** output *****/
#define _o_ss      plhs[0]

/* undocumented function prototype */
EXTERN_C mxArray *mxCreateSparseNumericMatrix(mwSize m, mwSize n, mwSize nzmax, mxClassID classid, mxComplexity ComplexFlag);

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    float *pr = NULL;
    mwIndex *ir = NULL, *jc = NULL;
    mwSize nzmax, m, n;
    double *tmp = NULL;
    int i;
	
    /* matrix size */
    nzmax = (mwSize)mxGetScalar(_i_nzmax);
    m     = (mwSize)mxGetScalar(_i_m);
    n     = (mwSize)mxGetScalar(_i_n);
	
    /* get pr */
    tmp = mxGetPr(_i_pr);
    pr = new float [nzmax];
    for (i=0; i<nzmax; i++) {
        pr[i] = (float)tmp[i];
    }

    /* get ir */
    tmp = mxGetPr(_i_ir);
    ir = new mwIndex [nzmax];
    for (i=0; i<nzmax; i++) {
        ir[i] = (mwIndex)tmp[i];
    }

    /* get jc */
    tmp = mxGetPr(_i_jc);
    jc = new mwIndex [n+1];
    for (i=0; i<n+1; i++) {
        jc[i] = (mwIndex)tmp[i];
    }
    
    /* make sparse matrix */
    _o_ss = mxCreateSparseNumericMatrix(m, n, nzmax, mxSINGLE_CLASS, mxREAL);
    memcpy((void*)mxGetPr(_o_ss), (const void*)pr, nzmax*sizeof(float));
    memcpy((void*)mxGetIr(_o_ss), (const void*)ir, nzmax*sizeof(mwIndex));
    memcpy((void*)mxGetJc(_o_ss), (const void*)jc, (n+1)*sizeof(mwIndex));
}
