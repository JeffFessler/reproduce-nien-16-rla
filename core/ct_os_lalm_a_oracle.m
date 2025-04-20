function [x,rmsd,pres,dres,rhok] = ct_os_lalm_a_oracle(...
    y,A,W,R,x0,...
    nIter,nBlock,alpha,s,delta,rho0,gamma0,eta,dwls,option,...
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

rho = rho0;
gamma = gamma0;

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

pres = zeros(nIter*nBlock,1);
dres = zeros(nIter*nBlock,1);
rhok = zeros(nIter*nBlock,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-a-oracle (nBlock = %g, alpha = %g, s = %g, delta = %g, rho0 = %g, gamma0 = %g, eta = %g, option = %s)...\\n',nBlock,alpha,s,delta,rho0,gamma0,eta,option);
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

        pp = norm((xbold-xb).*(rho*dwls)-(gbold-gb),1);
        dd = norm((gbold-gb)/rho-(gxbold-gxb),1);
        
        if pp>s*dd*delta          % primal residual is too large
            rho = rho*(1-gamma);  % decrease rho (increase primal step size)
            gamma = gamma*eta;
        elseif pp<s*dd/delta      % primal residual is too small
            rho = rho/(1-gamma);  % increase rho (increase dual step size)
            gamma = gamma*eta;
        end
        
        pres(nBlock*(iter-1)+k) = pp;
        dres(nBlock*(iter-1)+k) = dd;
        rhok(nBlock*(iter-1)+k) = rho;
        
        fprintf([back count],k);
        k = k+1;
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
rhok = [rho0; rhok];

x = vecx2matx(x);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
