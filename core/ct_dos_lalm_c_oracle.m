function [x,rmsd,ttoc] = ct_dos_lalm_c_oracle(...
    y,A,W,R,x0,...
    nIter,nBlock,alpha,dwls,p,option,...
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

a = @(t) single((pi/t*sqrt(1-(pi/2/t)^2))*(t>1)+(t<=1));

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

[count,back] = loop_count_str(nBlock);

rmsd0 = rms(x0-xref);

x = x0;
xb = x;

iblock = iOrder(end);
ia = iblock:nBlock:A.odim(end);
gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
gb = gxb;

switch option
    case {'max-curv','huber'}
        dreg = R.denom(R,zeros(np,1,'single'));
    otherwise
        exit('option is not available!?');
end

rmsd = zeros(nIter,1);
ttoc = zeros(nIter,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using DOS-LALM-c-oracle (nBlock = %g, alpha = %g, p = %g, option = %s)...\\n',nBlock,alpha,p,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
    
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for ib = 1:nBlock
        ii = (iter-1)*nBlock+ib;
    
        xt = (1-alpha)*x+alpha*xb;
        if mod(ii-1,p)==0
            xm = xt;
            greg = R.cgrad(R,xm);
            if strcmp(option,'huber')
                dreg = R.denom(R,xm);
            end
        end
        
        rho = a(ii);
        sb = rho*gxb+(1-rho)*gb;
        xb = proj(xb-(sb+greg+dreg.*(xt-xm))./(rho*dwls+alpha*dreg+eps));
        iblock = iOrder(ib);
        ia = iblock:nBlock:A.odim(end);
        gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
        gb = (rho*gxb+gb)/(rho+1);

        x = (1-alpha)*x+alpha*xb;
        
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
