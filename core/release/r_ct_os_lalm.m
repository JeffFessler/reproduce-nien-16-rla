function x = r_ct_os_lalm(...
    y,A,W,R,x0,...
    nIter,nBlock,nDenoise,thd,cst,rho,...
    xref,iROI,...
    iSave,sDir...
    )

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);

iSave = sort(unique(iSave));

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
iROI = matx2vecx(iROI);

nROI = sum(iROI);
rms = @(d) norm(d(iROI))/sqrt(nROI);

[gradi,iOrder,iFixed,dwls,dreg] = initialize_os(A,W,R,nBlock,thd,cst);

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

x = x0;
grad = gradi{iOrder(end)}(x,y);
g = grad;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM with continuation (nBlock = %g, nDenoise = %g, thd = %g, cst = %g, rho = %g)...\\n',nBlock,nDenoise,thd,cst,rho);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        xp = x-(grad+(1/rho-1)*g)./dwls;
        x = denoise_box_max_curv(x,rho*dwls,xp,R,dreg,iFixed,x0,nDenoise);
        grad = gradi{iblock}(x,y);
        g = (rho*grad+g)/(rho+1);

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
