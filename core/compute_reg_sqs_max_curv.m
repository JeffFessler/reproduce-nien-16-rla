function denom = compute_reg_sqs_max_curv(R,u)

C = R.C1;
absC = abs(C);
ck = absC*u;
% wt = pot.wpot(d);
switch R.pot_type
    case 'qgg2'
        wt = 1*compute_reg_wt(R);
    otherwise
        error 'max curvature is not available!!!';
end
denom = (absC'*(wt.*ck))./u;
