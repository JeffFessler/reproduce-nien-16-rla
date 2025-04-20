function rho = rhok(k,theta)
% function rho = rhok(k,theta) for theta \in [1,2]

mu_over_L = (pi./(2*k)).^2;

rho = ((mu_over_L.^2*theta^2+mu_over_L*theta-mu_over_L*theta^2+2*theta*(mu_over_L.*(1-mu_over_L).*(2*mu_over_L+theta-2*mu_over_L*theta)).^(1/2)-mu_over_L.^2)./(mu_over_L.^2*theta^2-2*mu_over_L.^2*theta+mu_over_L.^2-2*mu_over_L*theta^2+2*mu_over_L*theta+theta^2)).*(k>1)+(k<=1);