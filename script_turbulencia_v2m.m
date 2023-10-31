%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

Fs1 = 500;                 
dt1 = 1/Fs1;
T   = 60;               % total ength of signal (seconds)
AMP = 25;               % peak amplitude (mm) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t1 = 0:dt1:T;           
L = length(t1);

% random forcing
y1 = 2*(rand(length(t1),1)-0.5)';

dt2 = 1/8;
t2 = 0:dt2:(T-dt2);
y2 = interp1(t1,y1,t2,'cubic');
y2(1) = 0;
y2(end) = 0;

dt3 = 1/500;
t3 = 0:dt3:(T-dt3);
y3 = interp1(t2,y2,t3,'cubic');

figure(1);
hold on
plot(t3,y3,'k.-');
% hold off
plot(t2,y2,'r.-');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% NFFT1 = 2^nextpow2(length(y1)); % Next power of 2 from length of y
NFFT1 = length(y1);
Y1 = fft(y1,NFFT1)/length(y1); 
Fs1 = 1/dt1;
f1 = Fs1/2*linspace(0,1,NFFT1/2);

% NFFT2 = 2^nextpow2(length(y2)); % Next power of 2 from length of y
NFFT2 = length(y2);
Y2 = fft(y2,NFFT2)/length(y2); 
Fs2 = 1/dt2;
f2 = Fs2/2*linspace(0,1,NFFT2/2);

% NFFT3 = 2^nextpow2(length(y3)); % Next power of 2 from length of y
NFFT3 = length(y2);
Y3 = fft(y3,NFFT3)/length(y3); 
Fs3 = 1/dt3;
f3 = Fs3/2*linspace(0,1,NFFT3/2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot single-sided amplitude spectrum.
figure(2);
plot(f1,2*abs(Y1(1:NFFT1/2)),'b.-');
hold on;
plot(f2,2*abs(Y2(1:NFFT2/2)),'r.-');
plot(f3,2*abs(Y3(1:NFFT3/2)),'k.-'); 
legend('1','2','3')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f4 = f1;
temp=fft(y2);
temp=temp(1:round(end/2));
Y4 = [temp zeros(1,length(t1)-length(t2))];

hold on
plot(f4,2*abs(Y4(1:NFFT1/2)),'m.-'); 
legend('1','2','3','4')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y4 = real(ifft(Y4,length(f4)));

y4 = y4./max(abs(y4))*AMP;

figure(3);
hold on;
plot(t2,y2./max(abs(y2)),'b.-');
plot(t3,y3./max(abs(y3)),'k.-');
% plot(t1,y1,'k.-');
plot(linspace(0,T,length(y4)),y4,'m.-');
grid on
hold off

% csvwrite('turbsignalmodificada_01.csv',y4');
disp('CSV file exported');

figure
plot(linspace(0,2*T,2*length(y4)),[y4 y4],'m.-');

fmin = 7;
idx = find(abs(f4-fmin)<1e-2,1,'first');

Y5 = Y4;
Y5(1:idx) = zeros(idx,1); 

figure
plot(f4,2*abs(Y4(1:NFFT1/2)),'r:'); 
hold on;
plot(f4,2*abs(Y5(1:NFFT1/2)),'b--'); 
hold off
legend('pasabajo','pasabanda');


y5 = real(ifft(Y5,length(f4)));

y5 = y5./max(abs(y5))*AMP;


figure();
plot(linspace(0,T,length(y4)),y4,'r.-');
hold on
% plot(linspace(0,T,length(y5)),y5,'b.-');
legend('pasabajo','pasabanda');
grid on
hold off

disp(['Senal de duracion: ' num2str(T) ' seg'] )

csvwrite('turbsignal_60seg_Hasta-4Hz_AMP-25.csv',y4');
disp('CSV file exported');


