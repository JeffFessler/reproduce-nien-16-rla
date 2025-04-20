function x = ct_os_nes83_sag_mc(...
    y,A,W,R,x0,...
    nIter,nBlock,order,...
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

[iVals,Ai,gi,dwls] = setup_ct_sag1(A,W,nBlock,nIter,order);

dreg = R.denom(R,zeros(np,1,'single'));
dreg = dreg+0.1*min(dreg(dreg>0))*(dreg==0);

denom = dwls+dreg;

proj = @(x) max(x,0).*(dwls>0)+x0.*(dwls==0);

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

lstr = length(num2str(nBlock));
count = sprintf('[%%%dd/%d]',lstr,nBlock);
back = repmat('\b',[1 1+lstr+1+lstr+1]);

x = x0;
z = x;
xold = x;
told = 1;

d = zeros(np,1,'single');
cover = zeros(nBlock,1);
m = 0;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-Nes83 with max curvature (nBlock = %g, order = %s)...\\n',nBlock,order);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    k = 1;
    str.info = sprintf('Pass %g:',iter);
    fprintf([str.info ' ' count],0);
    str.log = strcat(str.log,str.info);
    for iblock = iVals((1:nBlock)+nBlock*(iter-1))
        if cover(iblock)==0
            cover(iblock) = 1;
            m = m+1;
        end
        
        ia = iblock:nBlock:A.odim(end);
        gnew = nBlock*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*z-col(y(:,:,ia))));
        d = d-Ai{iblock}'*(gi{iblock}-gnew);
        gi{iblock} = gnew;
        
        x = proj(z-(d/m+R.cgrad(R,z))./denom);
        
        t = (1+sqrt(1+4*told^2))/2;
        z = x+(told-1)/t*(x-xold);
        
        xold = x;
        told = t;

        fprintf([back count],k);
        k = k+1;
    end
    
    tt = toc;
    str.info = sprintf(' RMSD = %g (in %g seconds)\\n',rms(x-xref),tt);
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
