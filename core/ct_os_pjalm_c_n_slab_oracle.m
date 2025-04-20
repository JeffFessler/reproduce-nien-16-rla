function [x,rmsd] = ct_os_pjalm_c_n_slab_oracle(...
    y,A,W,R,x0,...
    nIter,nBlock,nSlab,option,...
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

[ct,bk] = loop_count_str(nSlab);
thickness = ceil(A.idim(3)/nSlab);
fprintf('Generate diagonal majorizing matrices for each slab... ');
dwls = zeros(np,1);
fprintf(ct,0);
for islab = 1:nSlab
    one = zeros(A.idim);
    one(:,:,1+(islab-1)*thickness:min(islab*thickness,A.idim(3))) = 1;
    one = one.*A.imask;
    % figure; im('mid3',one); cbar; pause;
    one = matx2vecx(one);
    dwls = dwls+(A'*(W*(A*one))).*one;
    fprintf([bk ct],islab);
end
% figure; im('mid3',vecx2matx(dwls)); cbar; pause;
fprintf('\n');

Ai = Gblock(A,nBlock,0);
iOrder = subset_start(nBlock)';
proj = @(x) max(x,0).*(1-iFBP)+x0.*iFBP;

a = @(t) single((pi/t*sqrt(1-(pi/2/t)^2))*(t>1)+(t<=1));

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

[count,back] = loop_count_str(nBlock);

x = x0;

t = 1;
rho = a(t);

iblock = iOrder(end);
ia = iblock:nBlock:A.odim(end);
gx = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*x-col(y(:,:,ia)))));
g = rho*gx;

gxold = gx;

switch option
    case {'max-curv','huber'}
        dreg = R.denom(R,zeros(np,1,'single'));
    otherwise
        exit('option is not available!?');
end

rmsd = zeros(nIter+1,1);
rmsd(1) = rms(x-xref);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-PJALM-c-oracle (nBlock = %g, option = %s)...\\n',nBlock,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    k = 1;
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for iblock = iOrder
        rho = a(t);
        
        ia = iblock:nBlock:A.odim(end);
        gx = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*x-col(y(:,:,ia)))));
        
        gxtra = 2*gx-gxold;
        s = rho*gxtra+(1-rho)*g;
        if strcmp(option,'huber')
            dreg = R.denom(R,x);
        end
        x = proj(x-(s+R.cgrad(R,x))./(rho*dwls+dreg+eps));
        g = (rho*gxtra+g)/(rho+1);
        
        gxold = gx;
        
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

x = vecx2matx(x);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
