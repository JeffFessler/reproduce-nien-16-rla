function [Ab,gradi,iOrder] = setup_ordered_subsets(A,W,nBlock)

Ab = Gblock(A,nBlock,0);

switch length(A.idim)
    case 2
        extract_views = @(x,ia) x(:,ia);
    case 3
        extract_views = @(x,ia) x(:,:,ia);
    otherwise
        error 'dimension is not applicable!!!';
end

for iblock = 1:nBlock
    ia = iblock:nBlock:W.idim(end);
    if strcmp(W.caller,'Gdiag')
        Wb{iblock} = Gdiag(extract_views(W.arg.diag,ia));
    elseif strcmp(W.caller,'Gweight')
        Wb{iblock} = Gweight(extract_views(W.arg.diag,ia),extract_views(W.arg.Kc,ia),W.arg.rho);
    else
        error 'unknown W.caller!!!';
    end
    gradi{iblock} = @(x,y) Ab{iblock}'*(Wb{iblock}*(Ab{iblock}*x-col(extract_views(y,ia))));
end

iOrder = subset_start(nBlock)';