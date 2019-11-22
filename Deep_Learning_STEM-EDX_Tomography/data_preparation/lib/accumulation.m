function dst = accumulation(src, num_avg)

bnd_avg      = 0:num_avg - 1;

sz          = size(src);
sz(3)       = sz(3) - num_avg + 1;

dst         = zeros(sz, 'like', src);

for iz = 1:sz(3)
    bnd_z                    	= iz + bnd_avg;
%     bnd_z(bnd_z > size(src, 3))	= [];
    
    dst(:,:,iz) = sum(single(src(:,:,bnd_z)), 3);
end