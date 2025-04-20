function [x,objv,rmsd] = ir_fast_gdm(x0,y,A,R,niter,xref)

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);
maty2vecy = @(maty) col(maty);
vecy2maty = @(vecy) reshape(vecy,A.odim);

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
np = length(x0);

y = maty2vecy(y);

grad = @(x) (A'*(A*x-y));
L = max(abs(A.arg.psf_fft(:)).^2);

obj = @(x) norm(y-A*x,2)^2/2+R.penal(R,x);
rms = @(d) norm(d)/sqrt(np);

x = x0;
z = x0;
xold = x;
told = 1;

objv = zeros(niter+1,1); objv(1) = obj(x);
rmsd = zeros(niter+1,1); rmsd(1) = rms(x-xref);

fprintf('Start solving image restoration problem using fast gradient descent method with restart...\n');
for iter = 1:niter
    x = z-(grad(z)+R.cgrad(R,z))./(L+R.denom(R,z));

    if (z-x)'*(x-xold)>0
        t = 1;
        z = x;
        fprintf('r');
    else
        t = 2/(1+sqrt(1+4/told^2));
        z = x+t*(1/told-1)*(x-xold);
        fprintf('.');
    end

    xold = x;
    told = t;
    
    objv(iter+1) = obj(x);
    rmsd(iter+1) = rms(x-xref);
    
    if mod(iter,100)==0
        fprintf(' (%g)\n',iter);
    end
end
if mod(iter,100)~=0
    fprintf(' (%g)\n',iter);
end

x = vecx2matx(x);