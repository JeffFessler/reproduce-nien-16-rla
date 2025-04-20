function u = design_d_minus_r1(beta,d,ubar)
% function u = design_d_minus_r1(beta,d,ubar) designs diagonal-minus-rank-one
% majorizer: beta * (D[d] - uu')

ubar = ubar/sum(abs(ubar));
g2 = min((1-1/beta)*d./abs(ubar));
u = sqrt(g2)*ubar;