function x = ct_high_mem_os_lalm_c_nrs0(...
    y,A,W,R,x0,...
    nIter,nBlock,eta,...
    xref,iROI,...
    iSave,sDir...
    )

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

a = @(k) single((pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1));

x = x0;
grad = gradi{iOrder(end)}(x,y)*nBlock;

g = grad;

C = R.C1;
pot = R.pot{1};
fprintf('Compute weighting matrix in R...\n');
tic;
wt = [];
for ii = 1:R.M
    wt = [wt; R.wt.col(ii)];
    fprintf('*');
end
tt = toc;
fprintf(' (Finish computing wt in R: in %g seconds)\n',tt);
thd = wt/single(eta);
% class(thd)
clear wt;

proj = @(x) max(x,0).*(dwls>0)+x0.*(dwls==0);
prox = @(z) pot.meth.shrink(pot,z,thd);
% prox = @(z) soft_shrink(z,thd);

Cx = C*x;
% v = Cx; e = 0*Cx;
v = prox(Cx); e = -Cx+v; % e = 0 initially

k = 1;
rho = a(k);
% class(rho)

str.info = sprintf('Start solving X-ray CT image reconstruction problem using high-memory OS-LALM with continuation [no restart] (nBlock = %g, eta = %g)...\\n',nBlock,eta);
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
        for ii = 1:length(v)
            z = Cx(ii)-e(ii);
            v(ii) = sign(z)*max(abs(z)-thd(ii),0);
        end
        e = e-Cx+v;
        
        k = k+1;
        rho = a(k);

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
