function x = ct_os_lalm_bb_vdc(...
    y,A,W,R,x0,...
    nIter,nBlock,nDenoise,...
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

[Ab,gradi,iOrder] = setup_ordered_subsets(A,W,nBlock);

fprintf('Compute the diagonal majorizer: dwls...\n');
dwls = compute_diag_majorizer(Ab,W,ones(size(x0)));

switch length(A.idim)
    case 2
        iInner = A.imask;
    case 3
        iInner = A.imask;
        nvz = squeeze(sum(sum(vecx2matx(dwls>0),1),2));
        nvm = sum(col(A.imask(:,:,floor(end/2))));
        iInner(:,:,nvz<nvm) = false;
        iInner = matx2vecx(iInner);
        clear nvz nvm;
    otherwise
        'dimension is not applicable!!!';
end

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

a = @(k) (pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1);

x = x0;
g = gradi{iOrder(end)}(x,y)*nBlock;
u = g;

k = 1;
l = 1;
rho = a(k)*iInner+a(l)*(1-iInner);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-BB with voxel-dependent continuation (nBlock = %g, nDenoise = %g)...\\n',nBlock,nDenoise);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
	    if iter==1 && iblock==iOrder(1)
            dsqs = dwls;
		    gold = g;
			xold = x;
        else
            dg = g-gold;
            dx = x-xold;
            dsqs = (min((dx'*dg)/(dx'*(dwls.*dx)),1)*iROI+(1-iROI)).*dwls;
            gold = g;
            xold = x;
        end
	    
        v = x-(g+(1./rho-1).*u)./(dsqs+eps);
        x = denoise_box(x,rho.*(dsqs+eps),v,R,dwls==0,x0,nDenoise);
        g = gradi{iblock}(x,y)*nBlock;
        u = (rho.*g+u)./(rho+1);
		
		k = k+1;
		rho = a(k)*iInner+a(l)*(1-iInner);

        str.info = sprintf('*');
        fprintf(str.info);
		str.log = strcat(str.log,str.info);
        if mod(iblock,100)==0
            str.info = sprintf('\\n');
            fprintf(str.info);
            str.log = strcat(str.log,str.info);
        end
    end
	l = l+1;
	rho = a(k)*iInner+a(l)*(1-iInner);
	
    tt = toc;
    str.info = sprintf(' (RMSD: %g) (%g: in %g seconds)\\n',rms(x-xref),iter,tt);
    fprintf(str.info);
	str.log = strcat(str.log,str.info);
    
    if sum(iSave==iter)>0
        fld_write([sDir 'x_iter_' num2str(iter) '.fld' ],vecx2matx(x));
    end
end

x = vecx2matx(x);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
