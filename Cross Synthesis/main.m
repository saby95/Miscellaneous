clear all, clc, close all

% load a .wav file
[x, fs] = audioread('carrier33.wav'); % get the samples of the .wav file
x = amplify(x);
x = x(:, 1);                        % get the first channel
xmax = max(abs(x));                 % find the maximum abs value
x = x/xmax;                         % scalling the signal

% define analysis parameters
xlen = length(x);                   % length of the signal
wlen = 1024;                        % window length (recomended to be power of 2)
h = wlen/4;                         % hop size (recomended to be power of 2)
nfft = 4096;                        % number of fft points (recomended to be power of 2)


% load a .wav file
[x1, fm] = audioread('modulator22.wav');% get the samples of the .wav file
%x1 = amplify(x1);
x1 = x1(:, 1);                        % get the first channel
x1max = max(abs(x1));                 % find the maximum abs value
x1 = x1/x1max;                         % scalling the signal

% define analysis parameters
xl1en = length(x1);                   % length of the signal
% define the coherent amplification of the window
K = sum(hamming(wlen, 'periodic'))/wlen;


[x2,fms] = audioread('vocodedsound.wav');
%figure, plot(x2)

%wlen = 4*wlen;
%w1len = 2*w1len;

[s, f, t] = stft(x, wlen, h, nfft, fs);
[s1, f1, t1] = stft(x1, wlen, h, nfft, fm);
%z = length(x)-length(x1);
%x1 = [x1;zeros(z,1)];
n = length(x)/length(t);
kg = n;
indx = 0; col = 1;
y1 = 0;
indx = 0; col = 1;
y2 = 0;
while indx + wlen <= xl1en
    % windowing
    temp = x(indx+1:indx+wlen);
    xw1(:,col) = temp;
    Y1(:,col) = envp(temp,fs)';
    temp1 = x1(indx+1:indx+wlen);
    xw2(:,col) = temp1;
    Y2(:,col) = envp(temp1,fm)';
    f1 = s(:,col)/Y1(:,col);
    Y(:,col) = f1*Y2(:,col);
    % update the indexes
    indx = indx + h;
    col = col + 1;
    disp(strcat('Processing time frame : ',int2str(col)));
end  
y = istft(Y,h,nfft,fm);
y = amplify(y);
y = amplify(y);
%figure
%plot(y);
obj = 1:1:400;
disp('The carrier sound is : ');
sound(x,fs);
pause(6);
disp('The modulator sound is : ');
sound(x1,fm);
pause(4);
disp('The cross synthesized sound is : ');
sound(y,fm);
audiowrite('output2.wav',y,fm);