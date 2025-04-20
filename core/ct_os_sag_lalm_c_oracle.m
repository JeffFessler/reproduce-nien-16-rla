function [x,rmsd,ttoc] = ct_os_sag_lalm_c_oracle(...
    y,A,W,R,x0,...
    nIter,nBlock,dwls,nReg,option,...
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
gzi = cell(1,nReg); sgz = zeros(np,1,'single');
dzi = cell(1,nReg); sdz = zeros(np,1,'single');
for ii = 1:nReg
    gzi{ii} = Ri{ii}.cgrad(Ri{ii},x);
    sgz = sgz+gzi{ii};
    switch option
        case 'max-curv'
            dzi{ii} = Ri{ii}.denom(Ri{ii},zeros(np,1,'single'))+eps;
        case 'huber'
            dzi{ii} = Ri{ii}.denom(Ri{ii},x)+eps;
        otherwise
            exit 'option is not available!?';
    end
    sdz = sdz+dzi{ii};
end

rmsd0 = rms(x-xref);

rmsd = zeros(nIter,1);
ttoc = zeros(nIter,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-SAG-LALM-c-oracle (nBlock = %g, nReg = %g, option = %s)...\\n',nBlock,nReg,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    fprintf(['Iter ' num2str(iter) ': ' count],0);
    for ib = 1:nBlock
        ii = (iter-1)*nBlock+ib;
        rho = a(ii);
        
        s = rho*gx+(1-rho)*g;
        x = proj(x-(s+sgz)./(rho*dwls+sdz));
        iblock = iOrder(ib);
        ia = iblock:nBlock:A.odim(end);
        gx = nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*x-col(y(:,:,ia)))));
        g = (rho*gx+g)/(rho+1);
        
        ir = randi(nReg);
        gzz = Ri{ir}.cgrad(Ri{ir},x);
        sgz = sgz-gzi{ir}+gzz; gzi{ir} = gzz;
        if strcmp(option,'huber')
            dzz = Ri{ir}.denom(Ri{ir},x);
            sdz = sdz-dzi{ir}+dzz; dzi{ir} = dzz;
        end
        
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
