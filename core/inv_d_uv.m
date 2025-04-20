function x = inv_d_uv(y,d,u,v)
% function x = inv_d_uv(y,d,u,v) solves the linear system: y =
% (D[d]+uv') x using the Sherman-Morrison formula

idu = u./d;
denom = 1+v'*idu;
if denom~=0
    idy = y./d;
    x = idy-(v'*idy)/denom*idu;
else
    error('system is not invertible?!');
end
