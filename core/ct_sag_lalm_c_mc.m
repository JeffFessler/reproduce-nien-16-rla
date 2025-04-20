function x = ct_sag_lalm_c_mc(...
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

[Ab,forwi,backi,iOrder] = setup_mini_batch_sag(A,W,nBlock);
forw = cell(1,nBlock);
for ii = 1:nBlock
    forw{ii} = zeros(Ab{ii}.size(1),1);
end
sgrad = zeros(np,1);

fprintf('Compute the diagonal majorizer: dwls...\n');
dwls = compute_diag_majorizer(Ab,W,ones(np,1));

fprintf('Compute the diagonal majorizer for the regularizer using max curvature: dreg...\n');
dreg = R.denom(R,zeros(np,1));

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

a = @(k) single((pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1));

x = x0;

forwnew = forwi{iOrder(end)}(x,y);
sgrad = sgrad+backi{iOrder(end)}(forwnew-forw{iOrder(end)}); forw{iOrder(end)} = forwnew;
grad = sgrad/nBlock;
g = grad;

k = 1;
rho = a(k);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using SAG-LALM (using max curvature) with continuation (nBlock = %g, nDenoise = %g)...\\n',nBlock,nDenoise);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        xp = x-(grad+(1/rho-1)*g)./(dwls+eps);
        x = denoise_box_max_curv(x,rho*(dwls+eps),xp,R,dreg,dwls==0,x0,nDenoise);
        forwnew = forwi{iblock}(x,y);
        sgrad = sgrad+backi{iblock}(forwnew-forw{iblock}); forw{iblock} = forwnew;
        grad = sgrad/nBlock;
        g = (rho*grad+g)/(rho+1);
		
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
