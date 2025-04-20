function x = ct_os_iso(...
    y,A,W,R,x0,...
    nIter,nBlock,...
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
dwls = compute_diag_majorizer(Ab,W,ones(np,1));

fprintf('Compute the diagonal majorizer for the regularizer using max curvature: dreg...\n');
dreg = compute_reg_sqs_max_curv(R,ones(np,1));

dsqs = dwls+dreg+eps;

tic;
fprintf('Initialize surrogate functions: z...\n');
common = R.cgrad(R,x0)-dsqs.*x0;
z = cell(1,nBlock);
zbar = zeros(np,1);
for iblock = 1:nBlock
    z{iblock} = gradi{iblock}(x0,y)*nBlock+common;
    zbar = zbar+z{iblock};
    
    fprintf('*');
	if mod(iblock,100)==0
	    fprintf('\n');
	end
end
zbar = zbar/nBlock;
tt = toc;
fprintf(' (Finish computing z: in %g seconds)\n',tt);

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

x = x0;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-ISO (nBlock = %g)...\\n',nBlock);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
    
    for iblock = iOrder
        znew = gradi{iblock}(x,y)*nBlock+R.cgrad(R,x)-dsqs.*x;
        zbar = zbar+(znew-z{iblock})/nBlock;
        
		x = max(-zbar./dsqs,0); x(dwls==0) = x0(dwls==0);
        z{iblock} = znew;
		
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