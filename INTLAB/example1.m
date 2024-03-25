clear
clf

% f = @(x)(sin(pi*x)-sin(x)); a = 0; b = 20;
f = @(x) 4./(1+x.^2); a=0; b=1;
fplot(f,[a,b],LineWidth=1.6)

% ApproxIncl = verifyquad(f,a,b)
% rad(ApproxIncl)

% intval('pi')/4

% e = 1e-12;
tic, Approx = quad(f,a,b), toc
tic, Incl = verifyquad(f,a,b), toc
rad(Incl)


e = 1e-12;
tic, Approx = quad(f,a,b,e), toc
tic, Incl = verifyquad(f,a,b,e), toc
rad(Incl)

e = 1e-15;
tic, Approx = quad(f,a,b,e), toc
tic, Incl = verifyquad(f,a,b,e), toc
rad(Incl)