function x = sherman_morrison_dr2(y,d,u,v,p,q)
% function x = sherman_morrison_dr2(y,d,p,q,u,v) solves the linear
% system: y = (D[d]+uv'+pq') x using the Sherman-Morrison formula

iAp = inv_d_uv(p,d,u,v);
% denom = 1+v'*iAp;  % this is wrong but faster...
denom = 1+q'*iAp;
if denom~=0
    iAy = inv_d_uv(y,d,u,v);
    x = iAy-(q'*iAy)/denom*iAp;
else
    error('system is not invertible?!');
end
