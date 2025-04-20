function x = denoise_box(x0,w,y,R,iFixed,xo,niter)

grad = @(x) (w.*(x-y)+R.cgrad(R,x));
dsqs = @(x) (w+R.denom(R,x));

x = x0;
z = x;
xold = x;
told = 1;
for iter = 1:niter
    x = max(z-grad(z)./dsqs(z),0); x(iFixed) = xo(iFixed);
    if (z-x)'*(x-xold)>0
        t = 1;
        z = x;
    else
        t = (1+sqrt(1+4*told^2))/2;
        z = x+(told-1)/t*(x-xold);
    end
    
    xold = x;
	told = t;
end
