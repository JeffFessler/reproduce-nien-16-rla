function x = ct_high_mem_os_lalm_nrs(...
    y,A,W,R,x0,...
    nIter,nBlock,rho,eta,...
    xref,iROI,...
    iSave,sDir...
    )

rho = single(rho);
eta = single(eta);

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);

iSave = sort(unique(iSave));

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
iROI = matx2vecx(iROI);

np = length(x0);
nROI = sum(iROI);
rms = @(d) norm(d(iROI))/sqrt(nROI);

[Ab,gradi,iOrder] = setup_ordered_subsets(A,W,nBlock);

fprintf('Compute the diagonal majorizer: dwls...\n');
dwls = compute_diag_majorizer(Ab,W,ones(np,1));

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

x = x0;
grad = gradi{iOrder(end)}(x,y)*nBlock;

g = grad*rho;

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

proj = @(x) max(x,0).*(dwls>0)+x0.*(dwls==0);
% prox = @(z) pot.meth.shrink(pot,z,thd);
prox = @(z) soft_shrink1(z,thd,R.nthread);

Cx = C*x;
% v = Cx; e = 0*Cx;
v = prox(Cx); e = -Cx+v; % e = 0 initially

str.info = sprintf('Start solving X-ray CT image reconstruction problem using high-memory OS-LALM [no restart] (nBlock = %g, rho = %g, eta = %g)...\\n',nBlock,rho,eta);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        x = proj(x-(rho*grad+(1-rho)*g+eta*C'*(Cx-v-e))./(rho*dwls+eta*(4*R.M)));
        grad = gradi{iblock}(x,y)*nBlock;
        Cx = C*x;
        g = (rho*grad+g)/(rho+1);
        % v = prox(Cx-e);
        soft_shrink_ip1(v,Cx-e,thd,R.nthread);
        e = e-Cx+v;
        
        str.info = sprintf('*');
        fprintf(str.info);
		str.log = strcat(str.log,str.info);
        if mod(iblock,100)==0
            str.info = sprintf('\\n');
            fprintf(str.info);
            str.log = strcat(str.log,str.info);
        end
    end
	
    tt = toc;
    str.info = sprintf(' (RMSD: %g) (%g: in %g seconds)\\n',rms(x-xref),iter,tt);
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
