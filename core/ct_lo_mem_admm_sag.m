function x = ct_lo_mem_admm_sag(...
    y,A,W,R,x0,...
    nIter,nBlock,nSAG,nDenoise,option,order,...
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

[iVals,Ai,gi,dwls] = setup_ct_sag1(A,W,nBlock,nIter,order);
prompt = ['{min, median, max} of dwls = {' num2str(min(dwls)) ', ' num2str(median(dwls)) ', ' num2str(max(dwls)) '} ' ...
          '(2*median(dwls)/nBlock = ' num2str(2*median(dwls)/nBlock) ')\n' ...
          'rho: '];
rho = single(input(prompt));

proj = @(x) max(x,0).*(dwls>0)+x0.*(dwls==0);
% dwls = dwls+0.1*min(dwls(dwls>0))*(dwls==0);

denom = dwls+rho;

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

lstr = length(num2str(nBlock));
count = sprintf('[%%%dd/%d]',lstr,nBlock);
back = repmat('\b',[1 1+lstr+1+lstr+1]);

x = x0;
g = zeros(np,1,'single');
cover = zeros(nBlock,1);
m = 0;
% u0 = x; d0 = 0;
u = con_denoise(x,rho,x,R,proj,option,nDenoise); d = -x+u; b = u+d;
fprintf('\n');

str.info = sprintf('Start solving X-ray CT image reconstruction problem using lo-mem ADMM with SAG (nBlock = %g, nSAG = %g, nDenoise = %g, option = %s, order = %s)...\\n',nBlock,nSAG,nDenoise,option,order);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    k = 1;
    str.info = sprintf('Pass %g:',iter);
    fprintf([str.info ' ' count],0);
    str.log = strcat(str.log,str.info);
    for iblock = iVals((1:nBlock)+nBlock*(iter-1))
        if cover(iblock)==0
            cover(iblock) = 1;
            m = m+1;
        end
        ia = iblock:nBlock:A.odim(end);
        gnew = nBlock*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*x-col(y(:,:,ia))));
        g = g-Ai{iblock}'*(gi{iblock}-gnew);
        gi{iblock} = gnew;
        x = proj((1-rho./denom).*x-(g/m-rho*b)./denom);
        
        fprintf([back count],k);
        k = k+1;
    end
    
    if mod(iter,nSAG)==0
        fprintf(' ');
        u = con_denoise(u,rho,x-d,R,proj,option,nDenoise);
        d = d-x+u;
        b = u+d;
    end
	
    tt = toc;
    str.info = sprintf(' RMSD = %g (in %g seconds)\\n',rms(x-xref),tt);
    fprintf(str.info);
    str.log = strcat(str.log,str.info);
    
    if sum(iSave==iter)>0
        fld_write([sDir 'x_iter_' num2str(iter) '.fld' ],vecx2matx(x));
    end
end

% output u to satisfy the box constraint
x = vecx2matx(u);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
