function x = denoise_box_max_curv(x0,w,y,R,dreg,iFixed,xo,niter)

grad = @(x) (w.*(x-y)+R.cgrad(R,x));
dsqs = w+dreg;

x = x0;
z = x;
xold = x;
told = 1;
for iter = 1:niter
    x = max(z-grad(z)./dsqs,0); x(iFixed) = xo(iFixed);
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