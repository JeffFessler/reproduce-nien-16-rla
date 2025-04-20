function [x,rmsd,ttoc] = ct_os_lalm_oracle1(...
    y,A,W,R,x0,...
    nIter,nBlock,alpha,rho,dwls,option,...
    xref,iROI,iFBP,...
    iSave,sDir...
    )

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);

iSave = sort(unique(iSave));

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
iROI = matx2vecx(iROI);
iFBP = matx2vecx(iFBP);

np = length(x0);
nROI = sum(iROI);
rms = @(d) norm(d(iROI))/sqrt(nROI);

Ai = Gblock(A,nBlock,0);
iOrder = subset_start(nBlock)';
proj = @(x) max(x,0).*(1-iFBP)+x0.*iFBP;

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

[count,back] = loop_count_str(nBlock);

x = x0;
xb = x;

iblock = iOrder(end);
ia = iblock:nBlock:A.odim(end);
grad = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
gb = grad;

dreg0 = R.denom(R,zeros(np,1,'single'));

rmsd0 = rms(x-xref);

rmsd = zeros(nIter,1);
ttoc = zeros(nIter,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-oracle (nBlock = %g, alpha = %s, rho = %s, option = %s)...\\n',nBlock,func2str(alpha),func2str(rho),option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for ib = 1:nBlock
        ii = (iter-1)*nBlock+ib;
        aii = alpha(ii); rii = rho(ii);
        
        xt = (1-aii)*x+aii*xb;
        
        sb = rii*grad+(1-rii)*gb;
        num = sb+R.cgrad(R,xt);
        if strcmp(option,'huber')
            den = rii*dwls+aii*R.denom(R,xt)+eps;
        else
            den = rii*dwls+aii*dreg0+eps;
        end
        xb = proj(xb-num./den);
        iblock = iOrder(ib);
        ia = iblock:nBlock:A.odim(end);
        grad = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
        gb = (rii*grad+gb)/(rii+1);
        
        x = (1-aii)*x+aii*xb;
        
        fprintf([back count],ib);
    end
	
    tt = toc;
    fprintf(' ');
    str.info = sprintf('RMSD = %g (%g: in %g seconds)\\n',rms(x-xref),iter,tt);
    fprintf(str.info);
    str.log = strcat(str.log,str.info);
    rmsd(iter) = rms(x-xref);
    ttoc(iter) = tt;
    
    if sum(iSave==iter)>0
        fld_write([sDir 'x_iter_' num2str(iter) '.fld' ],vecx2matx(x));
    end
end
rmsd = [rmsd0; rmsd];
ttoc = [0; ttoc]; ttoc = cumsum(ttoc);

x = vecx2matx(x);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);