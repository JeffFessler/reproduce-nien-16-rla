function theta = thetak(k,rho)

icn = min(pi./(2*k),1);
theta = (icn*(rho+1).*(icn*rho-icn-rho+2*sqrt((1-icn)*(1-rho))+2))./((icn*rho-icn-rho).^2);