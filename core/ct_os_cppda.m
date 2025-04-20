function x = ct_os_cppda(...
    y,A,W,R,x0,...
    nIter,nBlock,nDenoise,rho,theta,...
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

x = x0;
g = gradi{iOrder(end)}(x,y)*nBlock;
gbar = g;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-CPPDA (nBlock = %g, nDenoise = %g, rho = %g, theta = %g)...\\n',nBlock,nDenoise,rho,theta);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        x = denoise_box(x,rho*(dwls+eps),x-gbar/rho./(dwls+eps),R,dwls==0,x0,nDenoise);
        grad = gradi{iblock}(x,y)*nBlock;
        gnew = (rho*grad+g)/(rho+1);
        gbar = gnew+theta*(gnew-g);
        g = gnew;
        
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
