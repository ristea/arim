%%  Documentation
% This matlab script is made to generate a realistic data set
% for automotive radar interference, with a single source
% of interference ( ARIM ).
%
%%  Data set generator
clear all;
rng(707);

% Signals parameters limits
snr_limits = [5, 40];
sir_limits = -5;
slope_limits = [0, 1.5];
nr_samples = 50;

sb0_mat = zeros(1, 1024);
sb_mat = zeros(1, 1024);
amplitude_mat = zeros(1, 2048);
distance_mat = zeros(1, 2048);
info_mat = zeros(1, 4);

index = 1;
for i=1:1:nr_samples
    for snr = snr_limits(1):5:snr_limits(2)
        for sir = sir_limits:5:snr+5
            for interfer_coef = slope_limits(1):0.1:slope_limits(2)

                    nr_targets = randi([1,4], 1);
                    A = randi([1,100],1, nr_targets) / 100;
                    A(randi([1,nr_targets])) = 1;
                    teta = unifrnd(-pi,pi, 1, nr_targets);
                    complexA = A.*exp(1i*teta);
                    r = randi([2,95], 1, nr_targets);

                    [sb0, sb, label, distance_label] = gen_signal(snr, sir, interfer_coef, complexA, r);

                    sb0_mat(index, :) = sb0;
                    sb_mat(index, :) = sb;
                    amplitude_mat(index, :) = label;
                    distance_mat(index, :) = distance_label;
                    
                    % Adding information about signal
                    info_mat(index, :) = [1, snr, sir, interfer_coef];

                    index = index + 1;
            end
         end
    end
end

save('arim.mat', 'sb0_mat' , 'sb_mat', 'amplitude_mat', 'distance_mat', 'info_mat');



