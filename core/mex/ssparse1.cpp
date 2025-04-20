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
    
    /* output structure */
    const char *field_names[] = {"pr", "ir", "jc"};
    mwSize odim[] = {1, 1};
    _o_ss = mxCreateStructArray(2, odim, 3, field_names);
    int pr_field, ir_field, jc_field, m_field, n_field;
    pr_field = mxGetFieldNumber(_o_ss, "pr");
    ir_field = mxGetFieldNumber(_o_ss, "ir");
    jc_field = mxGetFieldNumber(_o_ss, "jc");
    
    mxArray *field_value;
    
    field_value = mxCreateNumericMatrix(nzmax, 1, mxSINGLE_CLASS, mxREAL);
    memcpy((void*)mxGetPr(field_value), (const void*)pr, nzmax*sizeof(float));
    mxSetFieldByNumber(_o_ss, 0, pr_field, field_value);
    
    field_value = mxCreateNumericMatrix(nzmax, 1, mxINT64_CLASS, mxREAL);  // mxClassID
    memcpy((void*)mxGetPr(field_value), (const void*)ir, nzmax*sizeof(mwIndex));
    mxSetFieldByNumber(_o_ss, 0, ir_field, field_value);
    
    field_value = mxCreateNumericMatrix(n+1, 1, mxINT64_CLASS, mxREAL);  // mxClassID
    memcpy((void*)mxGetPr(field_value), (const void*)jc, (n+1)*sizeof(mwIndex));
    mxSetFieldByNumber(_o_ss, 0, jc_field, field_value);
}
