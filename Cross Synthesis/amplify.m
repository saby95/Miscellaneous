function Y = amplify(x)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
max_signal = max(abs(x));
factor = 1/max_signal;
Y = x.*factor;

end

