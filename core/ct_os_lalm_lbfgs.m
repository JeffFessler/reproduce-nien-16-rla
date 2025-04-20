function [x,rmsd] = ct_os_lalm_lbfgs(...
    y,A,W,R,x0,...
    nIter,nBlock,nMemory,alpha,rho,dwls,eta,option,...
    xref,iROI,iFBP,...
    iSave,sDir...
    )

% H = (rho*dwls+alpha*dreg)*eta

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

rmsd = zeros(nIter+1,1);

x = x0;
xb = x;
rmsd(1) = rms(x-xref);

iblock = iOrder(end);
ia = iblock:nBlock:A.odim(end);
gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
gb = rho*gxb;

vb = proj(xb);
eb = -xb+vb;

switch option
    case {'max-curv','huber'}
        dreg = R.denom(R,zeros(np,1,'single'));
    otherwise
        exit('option is not available!?');
end
dd0 = (rho*dwls+alpha*dreg+eps)*(1+eta);

sk = cell(1,nMemory);
yk = cell(1,nMemory);
ak = zeros(nMemory,1);
bk = zeros(nMemory,1);
nCover = 0;
cur = 1;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-LBFGS (nBlock = %g, nMemory = %g, alpha = %g, rho = %g, eta = %g, option = %s)...\\n',nBlock,nMemory,alpha,rho,eta,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
    
    k = 1;
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for iblock = iOrder
        % update xt (auxiliary seq. 1)
        xt = (1-alpha)*x+alpha*xb;
        if strcmp(option,'huber')
            dreg = R.denom(R,xt);
            dd0 = (rho*dwls+alpha*dreg+eps)*(1+eta);
        end
        % update xb (auxiliary seq. 2)
        % -- compute search direction
        sb = rho*gxb+(1-rho)*gb;
        tb = ((rho*dwls+alpha*dreg+eps)*eta).*(xb-vb-eb);
        qq0 = sb+R.cgrad(R,xt)+tb;
        % -- compute l-bfgs update of xb
        xbold = xb;
        if nCover==0
            tau = 1;
            zz = qq0./dd0;
        else
            qq = qq0;
            idx = cur:-1:(cur-nCover+1);
            for ii = mod(idx-1,nMemory)+1
                ak(ii) = (sk{ii}'*qq)/(yk{ii}'*sk{ii});
                qq = qq-ak(ii)*yk{ii};
            end
            % tau = min((yk{cur}'*sk{cur})/(sk{cur}'*(dd0.*sk{cur})),1).*iROI+(1-iROI);
            tau = (yk{cur}'*sk{cur})/(sk{cur}'*(dd0.*sk{cur}));
            dd = tau.*dd0;
            zz = qq./dd;
            for ii = mod(idx(end:-1:1)-1,nMemory)+1
                bk(ii) = (yk{ii}'*zz)/(yk{ii}'*sk{ii});
                zz = zz+(ak(ii)-bk(ii))*sk{ii};
            end
            cur = mod(cur,nMemory)+1;
        end
        % xb = xb-zz;
        step = (1-0.25)*(zz'*qq0)/(zz'*(dd0.*zz)/2);
        xb = xb-step*zz;
        % step = 5e-1/((iter-1)*nBlock+k);
        % xb = xb-step*zz;
        nCover = min(nCover+1,nMemory);
        sk{cur} = xb-xbold;
        % -- compute gradient of xb
        gxbold = gxb;
        ia = iblock:nBlock:A.odim(end);
        gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
        yk{cur} = gxb-gxbold;
        % -- update split gradient
        gb = (rho*gxb+gb)/(rho+1);
        % -- update split/dual variable for the box constraint
        vb = proj(xb-eb);
        eb = eb-xb+vb;
        % update x (main seq.)
        x = (1-alpha)*x+alpha*xb;
        
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

x = vecx2matx(x);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
