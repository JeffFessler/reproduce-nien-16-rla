function x = ct_os_lalm_bb_c1(...
    y,A,W,R,x0,...
    nIter,nBlock,nDenoise,...
    xref,iROI,...
    iSave,sDir...
    )

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
iROI = matx2vecx(iROI);

np = length(x0);
nROI = sum(iROI);
rms = @(d) norm(d(iROI))/sqrt(nROI);

Ab = Gblock(A,nBlock,0);
dwls = zeros(np,1);

fprintf('Compute the diagonal majorizer: dwls...\n');
tic;
for iblock = 1:nBlock
    ia = iblock:nBlock:A.odim(3);
    if strcmp(W.caller,'Gdiag')
        Wb{iblock} = Gdiag(W.arg.diag(:,:,ia));
    elseif strcmp(W.caller,'Gweight')
        Wb{iblock} = Gweight(W.arg.diag(:,:,ia),W.arg.Kc(:,:,ia),W.arg.rho);
    else
        error 'unknown W.caller!!!';
    end
    gradi{iblock} = @(x) Ab{iblock}'*(Wb{iblock}*(Ab{iblock}*x-col(y(:,:,ia))));
	dwls = max(dwls,Ab{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ab{iblock}*ones(np,1))));
	
	fprintf('*');
	if mod(iblock,100)==0
	    fprintf('\n');
	end
end
dwls = dwls*nBlock;
tt = toc;
fprintf(' (Finish computing dwls: in %g seconds)\n',tt);

iOrder = subset_start(nBlock)';

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

a = @(k) (pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1);

x = x0;
g = gradi{iOrder(end)}(x)*nBlock;
u = g;

k = 1;
rho = a(k);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-BB with continuation (nBlock = %g, nDenoise = %g)...\\n',nBlock,nDenoise);
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
		
        v = x-(g+(1/rho-1)*u)./(dsqs+eps);
        x = denoise_box(x,rho*(dsqs+eps),v,R,dwls==0,x0,nDenoise);
        g = gradi{iblock}(x)*nBlock;
        u = (rho*g+u)/(rho+1);
		
		k = k+1;
		rho = a(k);

        str.info = sprintf('*');
        fprintf(str.info);
		str.log = strcat(str.log,str.info);
        if mod(iblock,100)==0
            str.info = sprintf('\\n');
            fprintf(str.info);
            str.log = strcat(str.log,str.info);
        end
    end
	
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

function x = denoise_box(x0,w,y,R,iFixed,xo,niter)

grad = @(x) (w.*(x-y)+R.cgrad(R,x));
dsqs = @(x) (w+R.denom(R,x));

x = x0;
z = x;
xold = x;
told = 1;
for iter = 1:niter
    x = max(z-grad(z)./dsqs(z),0); x(iFixed) = xo(iFixed);
    if (z-x)'*(x-xold)>0
        t = 1;
        z = x;
    else
        t = (1+sqrt(1+4*told^2))/2;
        z = x+(told-1)/t*(x-xold);
    end
    
    xold = x;
	told = t;
end