function [x,rmsd] = ct_sdca_succ_approx(...
    y,A,W,R,x0,...
    nIter,nBlock,nPeriod,...
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

dreg = R.denom(R,zeros(np,1,'single'))+eps;
den = cell(1,nBlock);
z = cell(1,nBlock);
fprintf(['Start computing denominator... ' count],0);
for iblock = 1:nBlock
    ia = iblock:nBlock:A.odim(end);
    nr = prod(A.odim(1:end-1))*length(ia);
    den{iblock} = (Ai{iblock}*((Ai{iblock}'*ones(nr,1))./dreg)).*col(W.arg.diag(:,:,ia))+1;
    z{iblock} = zeros(nr,1);
    fprintf([back count],iblock);
end
fprintf('\n');

b = x0-R.cgrad(R,x0)./dreg;
ii = 0;

AWz = zeros(A.np,1);
x = proj(b-AWz./dreg);

rmsd = zeros(nIter+1,1);
rmsd(1) = rms(x0-xref);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using SDCA with successive approximation (nBlock = %g, nPeriod = %g)...\\n',nBlock,nPeriod);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
    
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    kk = 0;
    for iblock = randi(nBlock,[1 nBlock])
    % for iblock = circshift(subset_start(nBlock),randi(nBlock))'
        ia = iblock:nBlock:A.odim(end);
        dz = (Ai{iblock}*x-col(y(:,:,ia))-z{iblock})./den{iblock};
        z{iblock} = z{iblock}+dz;
        AWz = AWz+Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*dz);
        x = proj(b-AWz./dreg);
        
        ii = ii+1;
        if ii>=nPeriod
            ii = 0;
            b = x-R.cgrad(R,x)./dreg;
            x = proj(b-AWz./dreg);
        end
        
        kk = kk+1;
        fprintf([back count],kk);
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
