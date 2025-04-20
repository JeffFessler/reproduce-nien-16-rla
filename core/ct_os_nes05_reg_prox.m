function x = ct_os_nes05_reg_prox(...
    y,A,W,R,x0,...
    nIter,nBlock,nDenoise,...
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
dwls = compute_diag_majorizer(Ab,W,ones(size(x0)));

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

z0 = x0;

x = x0;
z = z0;
v = x0;
t = 1;
sgrad = zeros(np,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-Nes05-reg-prox (nBlock = %g, nDenoise = %g)...\\n',nBlock,nDenoise);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        grad = gradi{iblock}(z,y)*nBlock;
        sgrad = sgrad+t*grad;
        
        %x = denoise_box(x,dwls+eps,z-grad./(dwls+eps),R,dwls==0,x0,nDenoise);
        %v = denoise_box(v,dwls+eps,z0-sgrad./(dwls+eps),R,dwls==0,x0,nDenoise);
        x = denoise_box(z,dwls+eps,z-grad./(dwls+eps),R,dwls==0,x0,nDenoise);
        v = denoise_box(z,dwls+eps,z0-sgrad./(dwls+eps),R,dwls==0,x0,nDenoise);
        
        t = (1+sqrt(1+4*t^2))/2;
        z = (1-1/t)*x+1/t*v;

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