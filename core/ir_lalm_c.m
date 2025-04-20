function [x,objv,rmsd] = ir_lalm_c(x0,y,A,R,niter,ndenoise,xref)

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

a = @(k) (pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1);

x = x0;
g = grad(x);
u = g;

k = 1;
rho = a(k);

objv = zeros(niter+1,1); objv(1) = obj(x);
rmsd = zeros(niter+1,1); rmsd(1) = rms(x-xref);

fprintf('Start solving image restoration problem using LALM with continuation and restart (nDenoise = %g)...\n',ndenoise);
for iter = 1:niter
    gold = g;
    
    v = x-(g+(1/rho-1)*u)/L;
    x = denoise_box(x,rho*L,v,R,false(np,1),x0,ndenoise);
    g = grad(x);
    u = (rho*g+u)/(rho+1);

    if (u-g)'*(g-gold)>0
        k = 1;
        u = g;
        fprintf('r');
    else
        k = k+1;
        fprintf('.');
    end
    rho = a(k);
    
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