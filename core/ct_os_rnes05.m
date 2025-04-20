function x = ct_os_rnes05(...
    y,A,W,R,x0,...
    nIter,nBlock,c,gamma,...
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
drlx = gamma*10^floor(log10(max(dwls)));

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

z0 = x0;

x = x0;
z = z0;
k = 0;
sgrad = zeros(np,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-rNes05 (nBlock = %g, c = %g, gamma = %g)...\\n',nBlock,c,gamma);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        grad = gradi{iblock}(z,y)*nBlock+R.cgrad(R,z);
        sgrad = sgrad+(k+1)/(2^c)*grad;
        dsqs = dwls+R.denom(R,z)+eps+(k+2)^c*drlx;
        
        x = max(z-grad./dsqs,0); x(dwls==0) = x0(dwls==0);
        v = max(z0-sgrad./dsqs,0); v(dwls==0) = x0(dwls==0);
        
        z = (k+1)/(k+3)*x+2/(k+3)*v;
        k = k+1;

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
