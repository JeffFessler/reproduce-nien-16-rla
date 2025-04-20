function u0 = initialize_un_factor(x0)

gxx0 = convn(x0,[1 0 -1; 2 0 -2; 1 0 -1]','same');
gyx0 = convn(x0,[1 2 1; 0 0 0; -1 -2 -1]','same');
gx0 = sqrt(gxx0.^2+gyx0.^2);
absx0 = abs(x0);
u0 = 2/3*(gx0/max(gx0(:)))+1/3*(absx0/max(absx0(:)));
