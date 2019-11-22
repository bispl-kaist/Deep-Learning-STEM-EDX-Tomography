function [dst, param] = registration(src, ref, param)
bparam = false;

if nargin == 3
    bparam = true;
end

dst      = zeros(size(src), 'like', src);

for ii = 1:size(ref, 3)

    if ~bparam
        [output Greg]	= dftregistration_v2(fft2(ref(:,:,1)), fft2(ref(:,:,ii)), fft2(src(:,:,ii)), 50);
    else
        output          = param(:,:,:,:,ii);
    end

    param(:,:,:,:,ii)   = output;
    dst(:,:,ii)         = circshift(src(:,:,ii),[round(output(3)) round(output(4))]);

end
