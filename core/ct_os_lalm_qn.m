function [x,rmsd] = ct_os_lalm_qn(...
    y,A,W,R,x0,...
    nIter,nBlock,alpha,rho,dwls,qn,eta,option,...
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

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-QN (nBlock = %g, alpha = %g, rho = %g, qn = %s, eta = %g, option = %s)...\\n',nBlock,alpha,rho,qn,eta,option);
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
        % update quasi-newton hessian
        if iter==1 && k==1
            dd = dd0;
            uu = zeros(A.np,1);
            vv = zeros(A.np,1);
            pp = zeros(A.np,1);
            qq = zeros(A.np,1);
        else
            % dd = dd0;
            % dd = min((dgxb'*dxb)/(dxb'*(dd0.*dxb)),1)*dd0;
            dd = (min((dgxb'*dxb)/(dxb'*(dd0.*dxb)),1).*iROI+(1-iROI)).*dd0;
            switch qn
                case 'sr1'
                    dd = dd0;
                    uu = dgxb-dd.*dxb;
                    vv = uu/(uu'*dxb);
                    pp = zeros(A.np,1);
                    qq = zeros(A.np,1);
                case 'broyden'
                    uu = dgxb-dd.*dxb;
                    vv = dxb/(dxb'*dxb);
                    pp = zeros(A.np,1);
                    qq = zeros(A.np,1);
                case 'bfgs'
                    uu = dgxb;
                    vv = uu/(dgxb'*dxb);
                    pp = dd.*dxb;
                    qq = -pp/(dxb'*pp);
                    % pp = zeros(A.np,1);
                    % qq = zeros(A.np,1);
                case 'bb'
                    uu = zeros(A.np,1);
                    vv = zeros(A.np,1);
                    pp = zeros(A.np,1);
                    qq = zeros(A.np,1);
                case 'none'
                    dd = dd0;
                    uu = zeros(A.np,1);
                    vv = zeros(A.np,1);
                    pp = zeros(A.np,1);
                    qq = zeros(A.np,1);
                otherwise
                    error('qn is not available?!');
            end
        end
        % update xb (auxiliary seq. 2)
        % -- compute search direction
        sb = rho*gxb+(1-rho)*gb;
        tb = ((rho*dwls+alpha*dreg+eps)*eta).*(xb-vb-eb);
        % -- compute qn update of xb
        dxb = xb;
        xb = xb-sherman_morrison_dr2(sb+R.cgrad(R,xt)+tb,dd,uu,vv,pp,qq);
        dxb = xb-dxb;
        % -- compute gradient of xb
        dgxb = gxb;
        ia = iblock:nBlock:A.odim(end);
        gxb = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*xb-col(y(:,:,ia)))));
        dgxb = gxb-dgxb;
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
