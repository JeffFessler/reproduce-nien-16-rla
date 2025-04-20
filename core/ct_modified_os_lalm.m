function [x,xb] = ct_modified_os_lalm(...
    y,A,W,R,x0,...
    nIter,nBlock,alpha,option,...
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

[iOrder,Ai,dwls] = setup_os(A,W,ones(np,1),nBlock);
proj = @(x) max(x,0).*(dwls>0)+x0.*(dwls==0);
% dwls = dwls+0.1*min(dwls(dwls>0))*(dwls==0);

a = @(t) single((pi/t*sqrt(1-(pi/2/t)^2))*(t>1)+(t<=1));
% b = @(t) single(alpha);
b = @(t) single(alpha*(t>1)+(t<=1));
% b = @(t) single(max(2/(t+1),alpha));
% b = @(t) single(max(1/t,alpha));

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
% gb = 0*gxb;
gb = gxb;

switch option
    case {'max-curv','huber'}
        dreg = R.denom(R,zeros(np,1,'single'));
    otherwise
        exit('option is not available!?');
end

t = 1;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using modified OS-LALM (nBlock = %g, alpha = %g, option = %s)...\\n',nBlock,alpha,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    k = 1;
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for iblock = iOrder
        rho = a(t);
        alp = b(t);
        
        xt = (1-alp)*x+alp*xb;
        if strcmp(option,'huber')
            dreg = R.denom(R,xt);
        end
        sb = rho*gxb+(1-rho)*gb;
        xb = proj(xb-(sb+R.cgrad(R,xt))./(rho*dwls+alp*dreg+eps));
        ia = iblock:nBlock:A.odim(end);
        gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
        gb = (rho*gxb+gb)/(rho+1);
        x = (1-alp)*x+alp*xb;
        
        fprintf([back count],k);
        
        k = k+1;
        t = t+1;
    end
	
    tt = toc;
    fprintf(' ');
    str.info = sprintf('RMSD = %g/%g (%g: in %g seconds)\\n',rms(x-xref),rms(xb-xref),iter,tt);
    fprintf(str.info);
    str.log = strcat(str.log,str.info);
    
    if sum(iSave==iter)>0
        fld_write([sDir 'x_iter_' num2str(iter) '.fld' ],vecx2matx(x));
        fld_write([sDir 'xb_iter_' num2str(iter) '.fld' ],vecx2matx(xb));
    end
end

x = vecx2matx(x);
xb = vecx2matx(xb);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
