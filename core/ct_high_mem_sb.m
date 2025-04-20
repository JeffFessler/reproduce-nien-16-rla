function x = ct_high_mem_sb(...
    y,A,W,R,x0,...
    nIter,nPCG,pType,eta,...
    xref,iROI,...
    iSave,sDir...
    )

eta = single(eta);

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);

iSave = sort(unique(iSave));

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
iROI = matx2vecx(iROI);
AWy = A'*W*y(:);
clear y;

np = length(x0);
nROI = sum(iROI);
rms = @(d) norm(d(iROI))/sqrt(nROI);

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

x = x0;

C = R.C1;
pot = R.pot{1};
fprintf('Compute weighting matrix in R...\n');
tic;
thd = [];
for ii = 1:R.M
    thd = [thd; R.wt.col(ii)/eta];
    fprintf('*');
end
tt = toc;
fprintf(' (Finish computing wt in R: in %g seconds)\n',tt);
tic;
fprintf('Construct precon for the inner ls image update...\n');
switch pType
    case 'none'
        P = 1;
    case 'circ'
        P = qpwls_precon('circ0',{A,W},sqrt(eta)*C,A.imask);
    case 'slice-circ'
        P = slice_circ_precon(A,W,sqrt(eta)*C,A.imask);
        % P = slice_circ_precon1(A,W,sqrt(eta)*C,A.imask,'xyz');
    otherwise
        P = 1;
end
tt = toc;
fprintf('Finish constructing precon for the inner ls image update in %g seconds.\n',tt);
B = A'*(W*A)+eta*(C'*C);

prox = @(z) soft_shrink1(z,thd,R.nthread);

Cx = C*x;
% v = Cx; e = 0*Cx;
v = prox(Cx); e = -Cx+v; % e = 0 initially

str.info = sprintf('Start solving X-ray CT image reconstruction problem using high-memory SB (nPCG = %g, pType = %s, eta = %g)...\\n',nPCG,pType,eta);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    x = qpwls_pcg2_no_warning(x,B,AWy+eta*C'*(v+e),0,'precon',P,'niter',nPCG,'isave','last');
    Cx = C*x;
    soft_shrink_ip1(v,Cx-e,thd,R.nthread);
    e = e-Cx+v;
    
    tt = toc;
    str.info = sprintf('* (RMSD: %g) (%g: in %g seconds)\\n',rms(x-xref),iter,tt);
    fprintf(str.info);
    str.log = strcat(str.log,str.info);
    
    if sum(iSave==iter)>0
        fld_write([sDir 'x_iter_' num2str(iter) '.fld' ],vecx2matx(x));
    end
end

x = vecx2matx(x);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
