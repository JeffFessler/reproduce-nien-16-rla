function x = ct_os_nes83(...
    y,A,W,R,x0,...
    nIter,nBlock,...
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
z = x;
xold = x;
told = 1;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-Nes83 (nBlock = %g)...\\n',nBlock);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        grad = gradi{iblock}(z,y)*nBlock+R.cgrad(R,z);
        dsqs = dwls+R.denom(R,z)+eps;
        
        x = max(z-grad./dsqs,0); x(dwls==0) = x0(dwls==0);
        
        if (z-x)'*(x-xold)>0
            t = 1;
            z = x;
            str.info = sprintf('r');
			fprintf(str.info);
			str.log = strcat(str.log,str.info);
        else
            t = (1+sqrt(1+4*told^2))/2;
            z = x+(told-1)/t*(x-xold);
            str.info = sprintf('*');
			fprintf(str.info);
			str.log = strcat(str.log,str.info);
        end
        xold = x;
        told = t;

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
