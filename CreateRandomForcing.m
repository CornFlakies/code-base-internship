% function out = CreateRandomForcing(A,fmax,T,NoPoints,filename,plot_flag)
clear all
close all

% T        :  duracion de la se?al (en sec)
% NoPoints :
if nargin==0,
    A = 28;
    fmax = 4;
    T = 30;
    NoPoints = 15000;
    filename = 'randomforcingdemo.csv';
end

% Begin

Fs = NoPoints/T;
f = Fs/2*linspace(0,1,NoPoints/2);

amp = zeros(size(f));
amp(2:find((f<fmax),1,'last')) = 1;

phase = 2*(rand(size(f))-0.5)*pi;

ft = amp.*exp(1i.*phase);
ft = [ft fliplr(conj(ft(2:end)))];

signal = ifft(ft)*length(f);
signal = signal./max(abs(signal));
signal = signal*A;

if plot_flag == 1,
    figure
    plot(signal,'.-')
    title('Signal')
    figure
    plot([fliplr(-f(2:end)) f] ,fftshift(abs(fft(signal))),'.-')
    title('Abs(FFT(Signal))');
end


csvwrite(filename,s');
