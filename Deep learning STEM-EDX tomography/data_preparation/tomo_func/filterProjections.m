
%======================================================================
function [p,H] = filterProjections(p_in, filter, d)

if nargin < 3
    d = 1;
end

p = p_in;

% Design the filter
len = size(p,1);
H = designFilter(filter, len, d);

if strcmpi(filter, 'none')
    return;
end

p(length(H),1)=0;  % Zero pad projections

% In the code below, I continuously reuse the array p so as to
% save memory.  This makes it harder to read, but the comments
% explain what is going on.

p = fft(p);               % p holds fft of projections

p = bsxfun(@times, p, H); % faster than for-loop

p = ifft(p,'symmetric');  % p is the filtered projections

p(len+1:end,:) = [];      % Truncate the filtered projections
%----------------------------------------------------------------------

