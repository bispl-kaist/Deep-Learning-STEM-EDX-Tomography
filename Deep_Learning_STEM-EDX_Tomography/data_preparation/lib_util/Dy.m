function [ y ] = Dy(x,bgpu)

if nargin < 2
    bgpu = true;
end

if bgpu
    x = gpuArray(x);
end

y = zeros(size(x), 'like', x);
y(1:end-1,:,:) = x(2:end,:,:);
y(end,:,:) = x(1,:,:);
y = y - x;

if bgpu
    y = gather(y);
end

end