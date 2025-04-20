function [x,objv,rmsd] = ir_adm_v_Cx(x0,y,A,R,eta,nPCG,niter,xref)

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);
maty2vecy = @(maty) col(maty);
vecy2maty = @(vecy) reshape(vecy,A.odim);

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
np = length(x0);

y = maty2vecy(y);

C = R.C1;
pot = R.pot{1};
prox = @(z,t) pot.meth.shrink(pot,z,t);
wt = R.wt.w(:);
P = qpwls_precon('circ0',{A,1},sqrt(eta)*C,A.imask);
B = A'*A+eta*(C'*C);

obj = @(x) norm(y-A*x,2)^2/2+R.penal(R,x);
rms = @(d) norm(d)/sqrt(np);

x = x0;
Cx = C*x;
v = Cx; e = 0*v;

v = prox(Cx-e,wt/eta); e = e-Cx+v;

objv = zeros(niter+1,1); objv(1) = obj(x);
rmsd = zeros(niter+1,1); rmsd(1) = rms(x-xref);

fprintf('Start solving image restoration problem using ADM with v = Cx split (eta = %g, nPCG = %g)...\n',eta,nPCG);
for iter = 1:niter
    x = qpwls_pcg2_no_warning(x,B,A'*y+eta*C'*(v+e),0,'precon',P,'niter',nPCG,'isave','last');
    Cx = C*x;
    
    v = prox(Cx-e,wt/eta); e = e-Cx+v;
    
    objv(iter+1) = obj(x);
    rmsd(iter+1) = rms(x-xref);
    
    fprintf('.');
    if mod(iter,100)==0
        fprintf(' (%g)\n',iter);
    end
end
if mod(iter,100)~=0
    fprintf(' (%g)\n',iter);
end

x = vecx2matx(x);
