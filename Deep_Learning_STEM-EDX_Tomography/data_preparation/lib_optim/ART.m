%%
% Âü°í¹®Çå 
% https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique

%%
function x  = ART(A,AT,b,x,lambda,n)

if (nargin < 6)
    n   = 1e2;
end

ATA	= AT(A(ones(size(x), 'single')));

for i = 1:n
    
    x  	= x + lambda*AT(b - A(x))./ATA;
    x(x < 0) = 0;
    
    figure(1); colormap gray;
    subplot(221);   imagesc(squeeze(x(:,:,floor(end/2))));      title(i);
    subplot(222);   imagesc(squeeze(x(:, floor(end/2), :)));
    subplot(223);   imagesc(squeeze(x(floor(end/2), :, :)));
    subplot(224);   imagesc(squeeze(b(:,:,floor(end/2))));
%     imagesc(squeeze(x(:,:)));      title(i);
    drawnow();
end

x   = gather(x);

end