function x = ct_os_lalm_nu_c(...
    y,A,W,R,x0,...
    nIter,nBlock,nDenoise,iUpdate,t,epsilon,...
    xref,iROI,...
    iSave,sDir...
    )

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);

iUpdate = sort(unique(iUpdate));
iSave = sort(unique(iSave));

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
iROI = matx2vecx(iROI);

np = length(x0);
nROI = sum(iROI);
rms = @(d) norm(d(iROI))/sqrt(nROI);

[Ab,gradi,iOrder] = setup_ordered_subsets(A,W,nBlock);

fprintf('Initialize the update-needed factor...\n');
un = matx2vecx(initialize_un_factor(vecx2matx(x0)));
unt = dr_adjustment(un,t,epsilon);

fprintf('Compute the diagonal majorizer: dwls...\n');
dwls = compute_diag_majorizer(Ab,W,unt);

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

a = @(k) (pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1);

x = x0;
xpre = x;
g = gradi{iOrder(end)}(x,y)*nBlock;
u = g;

k = 1;
rho = a(k);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-NU with continuation (nBlock = %g, nDenoise = %g, iUpdate = [%s], t = %g, epsilon = %g)...\\n',nBlock,nDenoise,num2str(iUpdate),t,epsilon);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        v = x-(g+(1/rho-1)*u)./(dwls+eps);
        x = denoise_box(x,rho*(dwls+eps),v,R,dwls==0,x0,nDenoise);
        g = gradi{iblock}(x,y)*nBlock;
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
    
    if sum(iUpdate==iter)>0
		un = abs(x-xpre);
        unt = dr_adjustment(un,t,epsilon);
		str.info = sprintf('Update the diagonal majorizer: dwls...\\n');
		fprintf(str.info);
		str.log = strcat(str.log,str.info);
		dwls = compute_diag_majorizer(Ab,W,unt);
    end
    
    if sum((iUpdate-1)==iter)>0
        xpre = x;
    end
end

x = vecx2matx(x);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
