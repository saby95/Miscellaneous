function rcepenvp = envp(x,fs)
Nframe = length(x); % frame size = 40 ms
w = hann(Nframe)';
x11 =rot90(x);
winspeech = x11 .* w;
Nfft = 4*Nframe; % factor of 4 zero-padding
sspec = fft(winspeech,Nfft);
dbsspecfull = 20*log(abs(sspec));
rcep = ifft(dbsspecfull);  % real cepstrum
rcep = real(rcep); % eliminate round-off noise in imag part
period = round(fs/200); % 41
nspec = Nfft/2+1;
aliasing = norm(rcep(nspec-10:nspec+10))/norm(rcep); % 0.02
nw = 2*period-4; % almost 1 period left and right
if floor(nw/2) == nw/2, nw=nw-1; end; % make it odd
w = boxcar(nw)'; % rectangular window
wzp = [w(((nw+1)/2):nw),zeros(1,Nfft-nw), ...
       w(1:(nw-1)/2)];  % zero-phase version
wrcep = wzp .* rcep.*aliasing;  % window the cepstrum ("lifter")
rcepenv = fft(wrcep); % spectral envelope
rcepenvp = real(rcepenv(1:nspec));
%rcepenvp = rcepenv;% should be real
rcepenvp = rcepenvp - mean(rcepenvp); % normalize to zero mean

end

