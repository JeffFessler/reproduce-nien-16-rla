function x = ct_os_aladmm_mc(...
    y,A,W,R,x0,...
    nIter,nBlock,rho,option,...
    xref,iROI,...
    iSave,sDir...
    )

matx2vecx = @(matx) matx(A.imask(:));
vecx2matx = @(vecx) embed(vecx,A.imask);

rho = single(rho);

iSave = sort(unique(iSave));

x0 = matx2vecx(x0);
xref = matx2vecx(xref);
iROI = matx2vecx(iROI);

np = length(x0);
nROI = sum(iROI);
rms = @(d) norm(d(iROI))/sqrt(nROI);
N = nIter*nBlock;

a = @(k) single((pi/k*sqrt(1-(pi/2/k)^2))*(k>1)+(k<=1));

[Ab,gradi,iOrder] = setup_ordered_subsets(A,W,nBlock);

fprintf('Compute the diagonal majorizer: dwls...\n');
dwls = compute_diag_majorizer(Ab,W,ones(np,1));

dreg = R.denom(R,zeros(np,1));

if sum(iSave==0)>0
    fld_write([sDir 'x_iter_0.fld' ],vecx2matx(x0));
end

switch option
    case 'thm2.5'
        alphat = @(t) 1;
        etat = @(t) rho*dwls+dreg;
        thetat = @(t) rho;
        taut = @(t) rho;
    case 'thm2.6'
        alphat = @(t) 2/(t+1);
        etat = @(t) rho*dwls+dreg*2/t;
        thetat = @(t) (t-1)/t*rho;
        taut = @(t) rho;
    case 'thm2.7'
        alphat = @(t) 2/(t+1);
        etat = @(t) rho*dwls*N/t+dreg*2/t;
        thetat = @(t) rho*N/t;
        taut = @(t) rho*N/t;
    case 'thm2.5c'
        alphat = @(t) 1;
        etat = @(t) a(t)*dwls+dreg;
        thetat = @(t) a(t);
        taut = @(t) a(t);
    case 'thm2.5cv'
        alphat = @(t) rho;
        etat = @(t) a(t)*dwls+rho*dreg;
        thetat = @(t) a(t);
        taut = @(t) a(t);
%     case 'thm2.5v'
%         alphat = @(t) 1;
%         etat = @(t) a(t)*dwls+rho*dreg;
%         thetat = @(t) a(t);
%         taut = @(t) a(t);
    case 'thm2.6c'
        alphat = @(t) 2/(t+1);
        etat = @(t) a(t)*dwls+dreg*2/t;
        thetat = @(t) (t-1)/t*a(t);
        taut = @(t) a(t);
    otherwise
        alphat = @(t) 1;
        etat = @(t) rho*dwls+dreg;
        thetat = @(t) rho;
        taut = @(t) rho;
end

proj = @(x) max(x,0).*(dwls>0)+x0.*(dwls==0);

t = 1;
x = x0;
xt = x;
gx = gradi{iOrder(end)}(x,y)*nBlock;
% g = thetat(t)*gx;
g = 0*gx;

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-ALADMM [max curvature] (nBlock = %g, rho = %g, option = %s)...\\n',nBlock,rho,option);
fprintf(str.info);
str.log = str.info;
for iter = 1:nIter
    tic;
	
    for iblock = iOrder
        alpha = alphat(t);
        eta = etat(t);
        theta = thetat(t);
        tau = taut(t);
        
        xb = (1-alpha)*xt+alpha*x;
        s = theta*gx+(1-theta)*g;
        
        x = proj(x-(s+R.cgrad(R,xb))./eta);
        xt = (1-alpha)*xt+alpha*x;
        
        gx = gradi{iblock}(x,y)*nBlock;
        g = (tau*gx+g)/(tau+1);
        
        t = t+1;

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
    str.info = sprintf(' (RMSD: %g) (%g: in %g seconds)\\n',rms(xt-xref),iter,tt);
    fprintf(str.info);
	str.log = strcat(str.log,str.info);
    
    if sum(iSave==iter)>0
        fld_write([sDir 'x_iter_' num2str(iter) '.fld' ],vecx2matx(xt));
    end
end

x = vecx2matx(xt);

fid = fopen([sDir 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
