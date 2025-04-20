function denom = compute_nu_denom_reg(R,u)

umat = embed(u,R.mask);

denom = 0;
for mm = 1:numel(R.C1s)
    Cm = R.C1s{mm};
    Cm = abs(Cm);
    ck = Cm*umat; % |C|*u
    wt = reshape(R.wt.col(mm),Cm.odim);
    tmp = Cm'*(wt.*ck);
    denom = denom+tmp;
end

denom = denom(R.mask(:))./u;
