function x = r_ct_os_nes05(...
    y,A,W,R,x0,...
    nIter,nBlock,thd,cst,...
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

[gradi,iOrder,iFixed,dwls,dreg] = initialize_os(A,W,R,nBlock,thd,cst);

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

z0 = x0;

x = x0;
z = z0;
t = 1;
sgrad = zeros(np,1);

dsqs = dwls+dreg;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-Nes05 (nBlock = %g, thd = %g, cst = %g)...\\n',nBlock,thd,cst);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        grad = gradi{iblock}(z,y)+R.cgrad(R,z);
        sgrad = sgrad+t*grad;
        
        x = max(z-grad./dsqs,0); x(iFixed) = x0(iFixed);
        v = max(z0-sgrad./dsqs,0); v(iFixed) = x0(iFixed);
        
        t = (1+sqrt(1+4*t^2))/2;
        z = (1-1/t)*x+1/t*v;

        str.info = sprintf('*');
        fprintf(str.info);
		str.log = strcat(str.log,str.info);
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