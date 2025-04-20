function [x,rmsd] = ct_os_lalm_bls(...
    y,A,W,R,x0,...
    nIter,nBlock,alpha,rho,dwls,beta,option,...
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

lstr = length(num2str(nBlock));
count = sprintf('[%%%dd/%d]',lstr,nBlock);
back = repmat('\b',[1 1+lstr+1+lstr+1]);

x = x0;
xb = x;

iblock = iOrder(end);
ia = iblock:nBlock:A.odim(end);
gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
gb = rho*gxb;

switch option
    case {'max-curv','huber'}
        dreg = R.denom(R,zeros(np,1,'single'));
    otherwise
        exit('option is not available!?');
end

rmsd = zeros(nIter+1,1);
rmsd(1) = rms(x-xref);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-BLS (nBlock = %g, alpha = %g, rho = %g, beta = %g, option = %s)...\\n',nBlock,alpha,rho,beta,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
    
    k = 1;
    nbls = 0;
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for iblock = iOrder
        xt = (1-alpha)*x+alpha*xb;
        if strcmp(option,'huber')
            dreg = R.denom(R,xt);
        end
        sb = rho*gxb+(1-rho)*gb;
        xbold = xb;
        gxbold = gxb;
        loop = true;
        while loop
            xb = proj(xb-(sb+R.cgrad(R,xt))./(rho*dwls+alpha*dreg+eps));
            ia = iblock:nBlock:A.odim(end);
            gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
            loop = (xb-xbold)'*(gxb-gxbold)>(xb-xbold)'*(dwls.*(xb-xbold));
            if loop
                dwls = dwls*beta;
                nbls = nbls+1;
            end
        end
        gb = (rho*gxb+gb)/(rho+1);
        x = (1-alpha)*x+alpha*xb;
        
        fprintf([back count],k);
        k = k+1;
    end    
    
    tt = toc;
    fprintf(' nbls = %g, ',nbls);
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
