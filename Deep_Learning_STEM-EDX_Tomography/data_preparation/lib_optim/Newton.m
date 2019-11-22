%%
% Âü°í¹®Çå 
% https://en.wikipedia.org/wiki/Newton%27s_method

%%
function x  = Newton(A0,A1,b,x,n)

if (nargin < 5)
    n   = 1e2;
end

A1_     = A1(x);
obj     = zeros(n, 1);

for i = 1:n
%     x   = x - A0(x, b)./A1(x);
    x   = x - A0(x, b)./A1_;
    x(x < 0) = 0;
   
%     obj(i)  = COST.function(x);
%     
%     figure(1);  colormap gray;
%     suptitle(COST.equation);
%     subplot(221);   imagesc(squeeze(x(:,:,floor(end/2))));
%     subplot(222);   imagesc(squeeze(x(:, floor(end/2), :)));
%     subplot(223);   imagesc(squeeze(x(floor(end/2), :, :)));
%     subplot(224);   semilogy(obj, '*-');    hold on;
%                     grid on; grid minor;
%     
%     drawnow();
end

x = gather(x);

end