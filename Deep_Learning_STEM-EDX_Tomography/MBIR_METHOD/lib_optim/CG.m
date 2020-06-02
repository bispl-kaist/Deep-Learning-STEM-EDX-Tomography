%% REFERENCE
% https://en.wikipedia.org/wiki/Conjugate_gradient_method

%%
function x  = CG(A,b,x,n)

if (nargin < 4)
    n   = 1e2;
end

% r       = b - A*x;
r       = b - A(x);
p       = r;

rsold   = r(:)'*r(:);

for i = 1:n
    %    Ap   = A*p;
    Ap   = A(p);
    a    = rsold/(p(:)'*Ap(:));
    
    x    = x + a*p;
    r    = r - a*Ap;
    
    rsnew= r(:)'*r(:);
    
    if (sqrt(rsnew) < eps)
        break;
    end
    
    p    = r + (rsnew/rsold)*p;
    rsold= rsnew;
end

x   = gather(x);

end