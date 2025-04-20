function [x,objv,rmsd] = ir_admm_u_Ax_v_Cx_cont_restart(x0,y,A,R,eta,nPCG,niter,xref)

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);
maty2vecy = @(maty) col(maty);
vecy2maty = @(vecy) reshape(vecy,A.odim);

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
np = length(x0);

y = maty2vecy(y);

a = @(k) (pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1);

C = R.C1;
pot = R.pot{1};
prox = @(z,t) pot.meth.shrink(pot,z,t);
wt = R.wt.w(:);

obj = @(x) norm(y-A*x,2)^2/2+R.penal(R,x);
rms = @(d) norm(d)/sqrt(np);

x = x0;
Ax = A*x;
Cx = C*x;
u = Ax;
v = Cx; e = 0*v;

k = 1;
rho = a(k);
P = qpwls_precon('circ0',{A,rho},sqrt(eta)*C,A.imask);
B = rho*A'*A+eta*(C'*C);

u = (u+rho*Ax)/(1+rho);
v = prox(Cx-e,wt/eta); e = e-Cx+v;

objv = zeros(niter+1,1); objv(1) = obj(x);
rmsd = zeros(niter+1,1); rmsd(1) = rms(x-xref);

fprintf('Start solving image restoration problem using ADMM with u = Ax and v = Cx splits (rho = cont., eta = %g, nPCG = %g, restart)...\n',eta,nPCG);
for iter = 1:niter
	Axold = Ax;

    x = qpwls_pcg2_no_warning(x,B,A'*(y-(1-rho)*u)+eta*C'*(v+e),0,'precon',P,'niter',nPCG,'isave','last');
    Ax = A*x;
    Cx = C*x;

    if (u-Ax)'*(Ax-Axold)>0
		k = 1;
		u = Ax;
		fprintf('r');
	else
		k = k+1;
		u = (u+rho*Ax)/(1+rho);
		fprintf('.');
	end
	rho = a(k);
    P = qpwls_precon('circ0',{A,rho},sqrt(eta)*C,A.imask);
	B = rho*A'*A+eta*(C'*C);
    
	v = prox(Cx-e,wt/eta);
	e = e-Cx+v;
    
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