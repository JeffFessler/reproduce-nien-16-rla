function [x,rmsd] = ct_os_sf_oracle(...
    y,A,W,R,x0,...
    nIter,nBlock,nDenoise,dwls,option,...
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

% WA1 = reshape(W*(A*ones(A.np,1)),A.odim);

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

[count1,back1] = loop_count_str(nBlock);
[count2,back2] = loop_count_str(nDenoise);

if 1
    dwls0 = dwls;
    
    dwls = zeros(np,1,'single');
    one = ones(np,1,'single');
    fprintf(['Construct diagonal majorizing matrix... ' count1],0);
    for iblock = 1:nBlock
        ia = iblock:nBlock:A.odim(end);
        dwls = max(dwls,Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*one)));
        fprintf([back1 count1],iblock);
    end
    fprintf('\n');
    dwls = dwls*nBlock;
    
    % figure; im('mid3',vecx2matx(div0(dwls,dwls0)),'dwls/dwls0'); cbar;
end

x = x0;

switch option
    case {'max-curv','huber'}
        dreg = R.denom(R,zeros(np,1,'single'));
    otherwise
        exit('option is not available!?');
end

z = x; xold = x; told = 1;

rmsd = zeros(nIter+1,1);
rmsd(1) = rms(x-xref);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OSSF-oracle (nBlock = %g, nDenoise = %g, option = %s)...\\n',nBlock,nDenoise,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
    
    fprintf(['Iter ' num2str(iter) ': ' count1],0);
    % forward
    u = z;
    k = 1;
    for iblock = iOrder
        ia = iblock:nBlock:A.odim(end);
        gwls = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*u-col(y(:,:,ia)))));
        u = u-div0(gwls,dwls);
        % u = u-gwls./(dwls+eps);
        % den = nBlock*(Ai{iblock}'*col(WA1(:,:,ia))); u = u-div0(gwls,den);
        
        fprintf([back1 count1],k);
        k = k+1;
    end

    fprintf([' ' count2],0);
    % backward
    l = 1;
    v = x; xxold = x; ttold = 1;
    for iiter = 1:nDenoise
        % the step size of the forward part is scaled by nBlock, so the
        % regularization force of the backward part should also be scaled
        % by nBlock!
        num = dwls.*(v-u)+nBlock*R.cgrad(R,v);
        if strcmp(option,'huber')
            dreg = R.denom(R,v);
        end
        den = dwls+nBlock*dreg;
        x = proj(v-div0(num,den));
        % x = proj(v-num./(den+eps));
        if (v-x)'*(x-xxold)>0
            tt = 1;
            v = x;
        else
            tt = (1+sqrt(1+4*ttold^2))/2;
            v = x+(tt-1)/ttold*(x-xxold);
        end
        xxold = x; ttold = tt;

        fprintf([back2 count2],l);
        l = l+1;
    end
    if 0%(z-x)'*(x-xold)>0
        t = 1;
        z = x;
    else
        t = (1+sqrt(1+4*told^2))/2;
        z = x+(t-1)/told*(x-xold);
    end
    xold = x; told = t;
    
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
