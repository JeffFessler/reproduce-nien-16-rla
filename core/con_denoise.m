function x = con_denoise(x0,w,y,R,proj,option,niter)

lstr = length(num2str(niter));
count = sprintf('[%%%dd/%d]}',lstr,niter);
back = repmat('\b',[1 1+lstr+1+lstr+1+1]);

grad = @(x) (w.*(x-y)+R.cgrad(R,x));

x = x0;
z = x;
xold = x;
told = 1;
switch option
    case 'huber'
        dsqs = w+R.denom(R,z);
    case 'max'
        dsqs = w+R.denom(R,0*z);
    otherwise
        dsqs = w+R.denom(R,0*z);
end
fprintf(['{denoise: ' count],0);
for iter = 1:niter
    x = proj(z-grad(z)./dsqs);
    if (z-x)'*(x-xold)>0
        t = 1;
        z = x;
    else
        t = (1+sqrt(1+4*told^2))/2;
        z = x+(told-1)/t*(x-xold);
    end
    
    xold = x;
	told = t;
    if strcmp(option,'huber')
        dsqs = w+R.denom(R,z);
    end
    fprintf([back count],iter);
end
