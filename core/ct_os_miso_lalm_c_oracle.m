function [x,rmsd,ttoc] = ct_os_miso_lalm_c_oracle(...
    y,A,W,R,x0,...
    nIter,nBlock,dwls,nReg,...
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

x = x0;

iblock = iOrder(end);
ia = iblock:nBlock:A.odim(end);
gx = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*x-col(y(:,:,ia)))));
g = gx;

Ri = Reg1block(R,nReg);
zi = cell(1,nReg);
gzi = cell(1,nReg);
sz = zeros(np,1,'single');
sgz = zeros(np,1,'single');
for ii = 1:nReg
    zi{ii} = x;
    gzi{ii} = Ri{ii}.cgrad(Ri{ii},zi{ii});
    sz = sz+zi{ii};
    sgz = sgz+gzi{ii};
end
dreg = R.denom(R,zeros(np,1,'single'))/nReg;

rmsd0 = rms(x-xref);

rmsd = zeros(nIter,1);
ttoc = zeros(nIter,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-MISO-LALM-c-oracle (nBlock = %g, nReg = %g)...\\n',nBlock,nReg);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for ib = 1:nBlock
        ii = (iter-1)*nBlock+ib;
        rho = a(ii);
        
        s = rho*gx+(1-rho)*g;
        x = proj((rho*dwls.*x-s+dreg.*sz-sgz)./(rho*dwls+nReg*dreg));
        iblock = iOrder(ib);
        ia = iblock:nBlock:A.odim(end);
        gx = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*x-col(y(:,:,ia)))));
        g = (rho*gx+g)/(rho+1);
        
        ir = randi(nReg);
        zz = x; gzz = Ri{ir}.cgrad(Ri{ir},x);
        sz = sz-zi{ir}+zz; zi{ir} = zz;
        sgz = sgz-gzi{ir}+gzz; gzi{ir} = gzz;
        
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
