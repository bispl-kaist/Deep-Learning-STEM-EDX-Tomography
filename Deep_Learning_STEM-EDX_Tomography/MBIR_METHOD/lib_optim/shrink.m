function dst = shrink(src, lambda)

dst = sign(src).*max(abs(src) - lambda, 0);

end

