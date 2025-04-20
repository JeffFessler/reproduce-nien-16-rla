function [x,rmsd] = ct_os_fista_oracle(...
    y,A,W,R,x0,...
    nIter,nBlock,dwls,option,...
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

switch option
    case {'max-curv','huber'}
        dreg = R.denom(R,zeros(np,1,'single'));
    otherwise
        exit('option is not available!?');
end

z = x; xold = x; told = 1;

rmsd = zeros(nIter+1,1);
rmsd(1) = rms(x-xref);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-FISTA-oracle (nBlock = %g, option = %s)...\\n',nBlock,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
    
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    % gradient
    x = z;
    k = 1;
    for iblock = iOrder
        ia = iblock:nBlock:A.odim(end);
        gwls = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*x-col(y(:,:,ia)))));
        greg = R.cgrad(R,x);
        if strcmp(option,'huber')
            dreg = R.denom(R,x);
        end
        num = gwls+greg;
        den = dwls+dreg;
        x = proj(x-div0(num,den));
        
        fprintf([back count],k);
        k = k+1;
    end
    % momentum
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
