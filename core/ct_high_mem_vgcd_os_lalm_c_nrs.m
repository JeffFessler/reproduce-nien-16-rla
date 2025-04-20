function x = ct_high_mem_vgcd_os_lalm_c_nrs(...
    y,A,W,R,x0,...
    nIter,nBlock,eta,gsize,...
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

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

a = @(k) (pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1);

x = x0;
grad = gradi{iOrder(end)}(x,y)*nBlock;

g = grad;

pot = R.pot{1};
fprintf('Compute weighting matrix in R...\n');
tic;
wt = cell(1,R.M);
for ii = 1:R.M
    wt{ii} = R.wt.col(ii);
    fprintf('*');
end
tt = toc;
fprintf(' (Finish computing wt in R: in %g seconds)\n',tt);

proj = @(x) max(x,0).*(dwls>0)+x0.*(dwls==0);
% prox = @(z) pot.meth.shrink(pot,z,wt/eta);
prox = @(z,a) pot.meth.shrink(pot,z,a);

% Cx = C*x;
% % v = Cx; e = 0*Cx;
% v = prox(Cx); e = -Cx+v; % e = 0 initially
C = R.C1;
Ci = cell(1,R.M);
Cit = cell(1,R.M);
Cx = cell(1,R.M);
v = cell(1,R.M);
e = cell(1,R.M);
sig = cell(1,R.M);
sigma = zeros(np,1);
fprintf('Initialize split difference images...\n');
tic;
for ii = 1:R.M
	Ci{ii} = @(x) vec(C.Cc{ii}*vecx2matx(x));
	Cit{ii} = @(y) matx2vecx(C.Cc{ii}'*reshape(y,size(A.imask)));
	Cx{ii} = Ci{ii}(x);
	v{ii} = prox(Cx{ii},wt{ii}/eta);
	e{ii} = -Cx{ii}+v{ii};
	sig{ii} = eta*Cit{ii}(Cx{ii}-v{ii}-e{ii});
	sigma = sigma+sig{ii};
	fprintf('*');
end
tt = toc;
fprintf(' (Finish initializing split difference images: in %g seconds)\n',tt);

k = 1;
rho = a(k);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using high-memory vGCD OS-LALM with continuation [no restart] (nBlock = %g, eta = %g)...\\n',nBlock,eta);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        di = randperm(R.M,gsize(iter));
        
        x = proj(x-(rho*grad+(1-rho)*g+sigma)./(rho*dwls+eta*(4*R.M)));
        grad = gradi{iblock}(x,y)*nBlock;
        g = (rho*grad+g)/(rho+1);
		
	for ii = 1:gsize(iter)
	    Cx{di(ii)} = Ci{di(ii)}(x);
	    v{di(ii)} = prox(Cx{di(ii)}-e{di(ii)},wt{di(ii)}/eta);
	    e{di(ii)} = e{di(ii)}-Cx{di(ii)}+v{di(ii)};
	    signew = eta*Cit{di(ii)}(Cx{di(ii)}-v{di(ii)}-e{di(ii)});
	    sigma = sigma-sig{di(ii)}+signew;
	    sig{di(ii)} = signew;
	end
        
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
