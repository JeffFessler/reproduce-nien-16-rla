function x = ct_os_lalm_0sr1_c(...
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
dwls = compute_diag_majorizer(Ab,W,ones(size(x0)));

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

a = @(k) (pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1);

x = x0;
g = gradi{iOrder(end)}(x,y)*nBlock;
u = g;

k = 1;
rho = a(k);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-LALM-0SR1 with continuation (nBlock = %g)...\\n',nBlock);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        if iter==1 && iblock==iOrder(1)
			dsqs = dwls;
		    gold = g;
			xold = x;
            
            d1 = 1./(dsqs+eps);
            u1 = zeros(size(x));
        else
		    dg = g-gold;
			dx = x-xold;
			dsqs = (min((dx'*dg)/(dx'*(dwls.*dx)),1)*iROI+(1-iROI)).*dwls;
            % dsqs = dwls;
			gold = g;
			xold = x;
            
            d1 = 1./(dsqs+eps);
            tmp = dx-d1.*dg;
            if tmp'*dg<=1e-8*norm(dg)*norm(tmp)
                u1 = zeros(size(x));
            else
                u1 = tmp/sqrt(tmp'*dg);
            end
        end
		
        v = rho*g+(1-rho)*u;
        b1 = x-(d1.*v+(u1'*v)*u1)/rho;
        dreg = R.denom(R,x)+eps;
        d2 = 1./(dreg/rho);
        b2 = x-R.cgrad(R,x)./dreg;
        
        u1tilde = u1./d1/sqrt(1+u1'*(u1./d1));
        dtilde = 1./(1./d1+1./d2);
        utilde = u1tilde.*dtilde/sqrt(1-u1tilde'*(u1tilde.*dtilde));
        ctilde = (b1./d1-(u1tilde'*b1)*u1tilde)+b2./d2;
        btilde = ctilde.*dtilde+(utilde'*ctilde)*utilde;
        x = prox_ct_with_sr1_inv_hessian(btilde,dtilde,utilde,x0,dwls==0);
        g = gradi{iblock}(x,y)*nBlock;
        
        u = (rho*g+u)/(rho+1);
        
        k = k+1;
		rho = a(k);

        if norm(u1)==0
            str.info = sprintf('*');
        else
            str.info = sprintf('r');
        end
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
