function [iOrder,Ai,dwls] = setup_os(A,W,u,nBlock)

Ai = Gblock(A,nBlock,0);

lstr = length(num2str(nBlock));
count = sprintf('[%%%dd/%d]',lstr,nBlock);
back = repmat('\b',[1 1+lstr+1+lstr+1]);

dwls = 0;
fprintf(sprintf('Generating block projection matrices with ordered subsets... %s',count),0);
tic;
for iblock = 1:nBlock
    ia = iblock:nBlock:A.odim(end);
    dwls = max(dwls,nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*u)))./u);
    
    fprintf([back count],iblock);
end
tt = toc;
fprintf(' (in %g seconds)\n',tt);

iOrder = subset_start(nBlock)';
