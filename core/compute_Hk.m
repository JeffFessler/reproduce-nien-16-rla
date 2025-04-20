function [dk,uk] = compute_Hk(d,sk,yk)

gamma = 0.8;
tau = (yk'*sk)/(yk'*(d.*yk));
dk = gamma*tau*d;
% dk = gamma*d;

tk = sk-dk.*yk;
if tk'*yk<=(1e-8)*norm(tk)*norm(yk)
    uk = zeros(size(tk));
else
    uk = tk/sqrt(tk'*yk);
end