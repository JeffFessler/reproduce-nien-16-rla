function [x,u,z,gap_itr,gap_erg,const] = rlalm_lasso(x0,u0,z0,A,y,R,niter,dA,alpha,rho,xh,uh,zh)

f = @(x,u) norm(y-u)^2/2+R.meth.penal(R,x);
fh = f(xh,uh);

gap = @(x,u) f(x,u)-fh-zh'*(A*x-u);

c1 = rho/2/alpha*(norm(sqrt(dA).*(x0-xh))^2-norm(A*(x0-xh))^2);
c2 = 1/2/alpha*(sqrt(rho)*norm(u0-uh)+1/sqrt(rho)*norm(z0-zh))^2;
const = c1+c2;

[count,back] = loop_count_str(niter);

x = x0; Ax = A*x;
u = u0;
z = z0;
h = dA.*x-A'*(Ax-y);

cumx = x;
cumu = u;

gap0 = gap(x,u);
gap_itr = nan(niter,1);
gap_erg = nan(niter,1);

overwrite = 1;

fprintf(sprintf('solving sparse linear regression using relaxed lalm... %s',count),0);
for iter = 1:niter
    % x-update
    s = rho*A'*(u-y+z/rho)+rho*h;
    x = R.meth.shrink(R,s/rho./dA,1/rho./dA);
    Ax = A*x;
    % u-update
    ru = alpha*Ax+(1-alpha)*u;
    u = (rho*ru+(y-z))/(rho+1);
    % z-update
    z = z-rho*(ru-u);
    % h-update
    h = alpha*(dA.*x-A'*(Ax-y))+(1-alpha)*h;
    
    % compute cumsum
    cumx = cumx+x;
    cumu = cumu+u;
    
    % compute gap value
    g = gap(x,u);
    if g<1e-15, overwrite = 0; end
    if overwrite, gap_itr(iter) = g; end
    gap_erg(iter) = gap(cumx/iter,cumu/iter);
    
    fprintf([back count],iter);
end
fprintf('\n');

gap_itr = [gap0; gap_itr];
gap_erg = [gap0; gap_erg];
