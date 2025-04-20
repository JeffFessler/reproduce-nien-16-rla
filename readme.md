
https://github.com/JeffFessler/reproduce-nien-16-rla

Code for reproducing the results in the paper
H. Nien and J. A. Fessler,
"Relaxed linearized algorithms for faster X-ray CT image reconstruction,"
IEEE Trans. Med. Imag., 35(4):1090-8, Apr. 2026
https://doi.org/10.1109/TMI.2015.2508780


Notes to future self

XCAT simulation results for paper and supplementary material
from
`/net/gladstone/z/hungnien/dataset/xcat_axial/`
see `out/xref_delta_*.fld`

makes sinogram:
`test_x_ray_ct_recon_xcat_axial_proj_data.m`
requires a 645MB xcat file:
`/n/ir62/z/hungnien/xcat_high_res/xtrue_hi.fld`
(todo: put on deep blue data?)

makes reference:
`test_x_ray_ct_recon_xcat_axial_generate_xref.m`

recon:
`test_x_ray_ct_recon_xcat_axial_os_rlalm.m`

clinical data is based on this case:
`ct53.10710.10` 600^2 x 222, na=3611
see
`/n/gladstone/z/hungnien/dataset/ct53.10710.10`
`/n/iv2/y2/hungnien/ct53.10710.10/`


todo:
routines are in:
`/net/gladstone/z/hungnien/myFcn/core/`

LASSO work for supplementary material
`/n/gladstone/z/hungnien/codes/toy_problems/l1_ls/`
`test_lasso_primal_dual_gap.m`
`rlalm_lasso.m`
