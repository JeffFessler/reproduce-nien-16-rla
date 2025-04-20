function Lmax = power_iter_lmax(R,b0,niter)

b = b0/norm(b0);

fprintf('Start computing the maximum eigenvalue of CDC: Lmax...\n');
for iter = 1:niter
	tic;
	CDCb = mult_cdc(R,b);
	Lmax = (b'*CDCb)/(b'*b);
	tt = toc;
	fprintf('Iteration %g: Lmax = %g (in %g seconds)\n',iter,Lmax,tt);
	
	b = CDCb/norm(CDCb);
end
fprintf('Finish computing the maximum eigenvalue of CDC: Lmax...\n')

function CDCb = mult_cdc(R,b)

CDCb = 0;
for mm = 1:R.M
	Cm = R.C1s{mm};
	ck = Cm*embed(b,R.mask);
	pot = R.pot{mm};
	wt = pot.wpot(0)*reshape(R.wt.col(mm),size(ck));
	CDCb = CDCb+Cm'*(wt.*ck);
end
CDCb = CDCb(R.mask(:));
	

% xmat = embed(x,R.mask);
% umat = embed(u,R.mask);
% 
% denom = 0;
% for mm = 1:numel(R.C1s)
% 	Cm = R.C1s{mm};
% 	d = Cm*xmat;
% 	Cm = abs(Cm);
% 	ck = Cm*umat; % |C|*u
% 	pot = R.pot{mm};
% 	wt = pot.wpot(d); % potential function Huber curvatures
% 	wt = wt.*reshape(R.wt.col(mm),size(wt));
% 	tmp = Cm'*(wt.*ck);
% 	denom = denom+tmp;
% end
% 
% denom = denom(R.mask(:))./u;
