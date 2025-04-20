function [gradi,iOrder,iFixed,dwls,dreg] = initialize_os(A,W,R,nBlock,thd,cst)

Ab = Gblock(A,nBlock,0);

switch length(A.idim)
    case 2
        extract_views = @(x,ia) x(:,ia);
    case 3
        extract_views = @(x,ia) x(:,:,ia);
    otherwise
        error 'dimension is not applicable!!!';
end

Wb = cell(nBlock,1);
gradi = cell(nBlock,1);
iOrder = subset_start(nBlock)';

np = A.arg.np;
one = ones(np,1);
dmax = zeros(np,1);
dmin = inf(np,1);

fprintf('Initialize the ordered-subsets method...\n');
tic;
for iblock = 1:nBlock
    ia = iblock:nBlock:W.idim(end);
    
    if strcmp(W.caller,'Gdiag')
        Wb{iblock} = Gdiag(extract_views(W.arg.diag,ia));
    elseif strcmp(W.caller,'Gweight')
        Wb{iblock} = Gweight(extract_views(W.arg.diag,ia),extract_views(W.arg.Kc,ia),W.arg.rho);
    else
        error 'unknown W.caller!!!';
    end
    gradi{iblock} = @(x,y) Ab{iblock}'*(Wb{iblock}*(Ab{iblock}*x-col(extract_views(y,ia))))*nBlock;
    
    dwlsi = Ab{iblock}'*(col(extract_views(W.arg.diag,ia)).*(Ab{iblock}*one));
    dmax = max(dmax,dwlsi);
    dmin = min(dmin,dwlsi);
    
    fprintf('*');
	if mod(iblock,100)==0
	    fprintf('\n');
	end
end
iFixed = (dmax==0);
iCorr = (dmin./(dmax+iFixed))<thd;
dwls = dmax*nBlock;
dwls(iCorr) = max(dwls(iCorr),cst*max(dwls));
tt = toc;
fprintf(' (Finish computing dwls: in %g seconds)\n',tt);

dreg = R.denom(R,zeros(np,1));