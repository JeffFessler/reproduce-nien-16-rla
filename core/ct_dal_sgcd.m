function [x,rmsd] = ct_dal_sgcd(...
    y,A,W,R,x0,...
    nIter,nBlock,lambda,pDual,nFISTA,option,...
    xref,iROI,iFBP,...
    iSave,sDir...
    )

% rng(0);

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
nBlock = min(nBlock,A.odim(end));

Ai = Gblock(A,nBlock,0);
proj = @(x) max(x,0).*(1-iFBP)+x0.*iFBP;

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

[count,back] = loop_count_str(nBlock);
[count2,back2] = loop_count_str(nFISTA);

den = cell(1,nBlock);
z = cell(1,nBlock);
tic;
fprintf(['Start computing denominator... ' count],0);
for iblock = 1:nBlock
    ia = iblock:nBlock:A.odim(end);
    nr = prod(A.odim(1:end-1))*length(ia);
    den{iblock} = (Ai{iblock}*(lambda.*(Ai{iblock}'*ones(nr,1)))).*col(W.arg.diag(:,:,ia))+1;
    z{iblock} = zeros(nr,1);
    fprintf([back count],iblock);
end
tt = toc;
fprintf(' (0: in %g seconds)\n',tt);

x = x0;
AWz = zeros(A.np,1);

rmsd = zeros(nIter+1,1);
rmsd(1) = rms(x-xref);

xold = x;

fprintf('Perform image denoising: ');
x = denoise_fista(x,x-lambda.*AWz,1./lambda,R,proj,nFISTA,option);
fprintf('\n');

xbar = 2*x-xold;
b = AWz+xbar./lambda;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using DAL-SGCD (nBlock = %g, pDual = %g, nFISTA = %g, option = %s)...\\n',nBlock,pDual,nFISTA,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
    
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    
    kk = 0;
    for iblock = randi(nBlock,[1 nBlock])
    % for iblock = circshift(subset_start(nBlock),randi(nBlock))'
        ia = iblock:nBlock:A.odim(end);
        dz = (Ai{iblock}*(lambda.*(b-AWz))-col(y(:,:,ia))-z{iblock})./den{iblock};
        z{iblock} = z{iblock}+dz;
        AWz = AWz+Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*dz);
        
        kk = kk+1;
        fprintf([back count],kk);
    end
    
    if mod(iter,pDual)==0
        xold = x;
        
        fprintf(' ');
        x = denoise_fista(x,x-lambda.*AWz,1./lambda,R,proj,nFISTA,option);
        
        xbar = 2*x-xold;
        b = AWz+xbar./lambda;
    end
    
    tt = toc;
    fprintf(' ');
    str.info = sprintf('RMSD = %g (%g: in %g seconds)\\n',rms(x-xref),iter,tt);
    fprintf(str.info);
    str.log = strcat(str.log,str.info);
    rmsd(iter+1) = rms(x-xref);
    
    if sum(iSave==iter)>0
        fld_write([sDir 'x_iter_' num2str(iter) '.fld' ],vecx2matx(x));
    end
end

x = vecx2matx(x);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);

function x = denoise_fista(x0,y,w,R,proj,niter,option)
[ct,bk] = loop_count_str(niter);
dd = w+R.denom(R,zeros(size(x0),'single'))+eps;
z = x0; xold = x0; told = 1;
fprintf(ct,0);
for iter = 1:niter
    nn = (z-y).*w+R.cgrad(R,z);
    if strcmp(option,'huber')
        dd = w+R.denom(R,z)+eps;
    end
    x = proj(z-nn./dd);
    if (z-x)'*(x-xold)>0
        t = 1;
        z = x;
    else
        t = (1+sqrt(1+4*told^2))/2;
        z = x+(told-1)/t*(x-xold);
    end
    xold = x; told = t;
    fprintf([bk ct],iter);
end