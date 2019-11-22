function [ y ] = DyT(x,bgpu)

if nargin < 2
    bgpu = true;
end

if bgpu
    x = gpuArray(x);
end

y = zeros(size(x), 'like', x);
y(1:end-1,:,:) = x(2:end,:,:);
y(end,:,:) = x(1,:,:);
tempt = -(y - x);
difft = tempt(1:end-1,:,:);
y(2:end,:,:) = difft;
y(1,:,:) = x(end,:,:) - x(1,:,:);

if bgpu
    y = gather(y);
end

end