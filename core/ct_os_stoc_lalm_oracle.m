function [x,rmsd] = ct_os_stoc_lalm_oracle(...
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

Ri = Reg1block(R);
nr = length(Ri);
switch option
    case {'max-curv','huber'}
        dreg = R.denom(R,zeros(np,1,'single'));
    otherwise
        exit('option is not available!?');
end

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

[count,back] = loop_count_str(nBlock);

x = x0;
xb = x;

iblock = iOrder(end);
ia = iblock:nBlock:A.odim(end);
gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
gb = rho*gxb;

rmsd = zeros(nIter+1,1);
rmsd(1) = rms(x-xref);

l = 1;
% C = nBlock*nIter; eta = @(l) (sqrt(l)+C-1)/C;
b = 1.01; c = (b-1)/(sqrt(nBlock*nIter)-1); eta = @(l) c*sqrt(l)+(1-c);
% eta = @(l) 1;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-Stoc-LALM-oracle (nBlock = %g, alpha = %g, rho = %g, option = %s)...\\n',nBlock,alpha,rho,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    k = 1;
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for iblock = iOrder
        ii = randi(nr);
        xt = (1-alpha)*x+alpha*xb;
        sb = rho*gxb+(1-rho)*gb;
        if strcmp(option,'huber')
            dreg = nr*Ri{ii}.denom(Ri{ii},xt);
        end
        xb = proj(xb-(sb+nr*Ri{ii}.cgrad(Ri{ii},xt))./(rho*dwls+alpha*dreg*eta(l)+eps));
        ia = iblock:nBlock:A.odim(end);
        gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
        gb = (rho*gxb+gb)/(rho+1);
        x = (1-alpha)*x+alpha*xb;
        
        fprintf([back count],k);
        k = k+1;
        l = l+1;
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
