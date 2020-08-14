function [sb0, sb, label, distance_label] = gen_signal(snr,sir, interfer_coef, A1, r)

% Interference central frequency
fc = 15e6; 

%FMCW radar parameters
Tr=25.6e-6; % Period
bw=1.6e+9; % Bandwidth
slope=bw/Tr; % slope(Hz/s)
c0=3e+8; 

Fs = 40e6; % sampling frequency
N = Fs*Tr; % number of samples

t=0:1/Fs:Tr-1/Fs; %intervalul de timp(s) al semnalului transmis

Nfft=2^(nextpow2(2*N)); % number of FFT points

F = (0:1/Nfft:1-1/Nfft)*Fs; % frequency vector
r_axis = c0/2/slope*F;

ntarget = length(A1); % number of targets
t_d = 2*r/c0; % range to delay

%Transmitted signal
st=exp(1j*pi*slope*((t-Tr/2).^2));

sb0 = zeros(1,N);
% Received beat signal
for i = 1:ntarget  
    sb0 = sb0 + A1(i)*st.*exp(-1j*pi*slope*((t-Tr/2-t_d(i)).^2));
end
% Add complex Gaussian noise
sb0 = sb0 + 10^(-snr/20)/sqrt(2)*A1(1)*(randn(1,N)+1j*randn(1,N))*sqrt(N);

% Add interference    
f_cw_inst = -fc + (1-interfer_coef)*slope*(t - Tr/2);   % Instantaneous frequency

Acw = A1(1).*10^(-sir/20).*sqrt(slope*abs(1-interfer_coef))*N/Fs;   
cwt = Acw*st.*exp(-1j*2*pi*(fc*t + 0.5*(interfer_coef*slope)*(t - Tr/2).^2));

cwt(abs(f_cw_inst) > Fs/2) = 0;

sb = sb0 + cwt;

[label, distance_label] = get_label(r_axis, r, A1);

end

