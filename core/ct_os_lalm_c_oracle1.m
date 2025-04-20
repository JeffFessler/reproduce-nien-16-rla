function [x,rmsd,pres,dres,rhok] = ct_os_lalm_c_oracle1(...
    y,A,W,R,x0,...
    nIter,nBlock,alpha,dwls,option,...
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

lstr = length(num2str(nBlock));
count = sprintf('[%%%dd/%d]',lstr,nBlock);
back = repmat('\b',[1 1+lstr+1+lstr+1]);

if 0
    dwls0 = dwls;
    
    dwls = zeros(np,1,'single');
    one = ones(np,1,'single');
    fprintf(['Construct diagonal majorizing matrix... ' count],0);
    for iblock = 1:nBlock
        ia = iblock:nBlock:A.odim(end);
        dwls = max(dwls,Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*one)));
        fprintf([back count],iblock);
    end
    fprintf('\n');
    dwls = dwls*nBlock;
    
    % figure; im('mid3',vecx2matx(div0(dwls,dwls0)),'dwls/dwls0'); cbar;
end

x = x0;
xb = x;

t = 1;
rho = a(t);

iblock = iOrder(end);
ia = iblock:nBlock:A.odim(end);
gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
gb = rho*gxb;

switch option
    case {'max-curv','huber'}
        dreg = R.denom(R,zeros(np,1,'single'));
    otherwise
        error('option is not available!?');
end

rmsd = zeros(nIter+1,1);
rmsd(1) = rms(x-xref);

pres = zeros(nIter*nBlock,1);
dres = zeros(nIter*nBlock,1);
rhok = zeros(nIter*nBlock,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-c-oracle (nBlock = %g, alpha = %g, option = %s)...\\n',nBlock,alpha,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
    
    k = 1;
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for iblock = iOrder
        xbold = xb;
        gbold = gb;
        gxbold = gxb;
        
        rho = a(t);
        xt = (1-alpha)*x+alpha*xb;
        if strcmp(option,'huber')
            dreg = R.denom(R,xt);
        end
        sb = rho*gxb+(1-rho)*gb;
        xb = proj(xb-(sb+R.cgrad(R,xt))./(rho*dwls+alpha*dreg+eps));
        ia = iblock:nBlock:A.odim(end);
        gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
        gb = (rho*gxb+gb)/(rho+1);
        x = (1-alpha)*x+alpha*xb;
        
        pres(nBlock*(iter-1)+k) = norm((xb-xbold).*(rho*dwls)-(gb-gbold),2)^2;
        dres(nBlock*(iter-1)+k) = norm((gb-gbold)/rho-(gxb-gxbold),2)^2;
        rhok(nBlock*(iter-1)+k) = rho;
        
        fprintf([back count],k);
        k = k+1;
        t = t+1;
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
pres = [0; pres];
dres = [0; dres];
rhok = [rhok; a(t)];

x = vecx2matx(x);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
