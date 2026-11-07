% generate_dataset_v36_optimized.m
% -------------------------------------------------------------------------
% V3.6 OPTIMIZED: 4 Pilots with Robust Ridge LS
%
% Optimization:
%   - Np_pilot = 4 (Reduced)
%   - Ridge Regularization = 0.1 (Robust against interference)
% -------------------------------------------------------------------------

clc; clear; close all;

if ~isdeployed
    addpath('./useful_function/');
end

%% System params
global Nc Ns fC c Delta_f Ts Te;

c        = 3e8;
Delta_f  = 78.125e3;
Nc       = 1024;
Ns       = 512;
fC       = 5e9;
Te       = 1/Delta_f;
Ts       = Te;

delta_r  = 1.1574;
r_max    = 2500;
v_max    = 75;

n = 0:Nc-1;
k = n';
DFT_matrix = 1/sqrt(Nc) * exp(-2i * pi * k * n / Nc);

%% Dataset config
numSamples  = 20000;  % Increased for better generalization
hcom_length = 4;

% Feature config
B = 64;
bandSize = Nc / B;
percentiles = [10 25 50 75 90];
G = 4 + numel(percentiles);
xDim = 3*B + G + 1 + 2*hcom_length;  % 210
yDim = 2*hcom_length;

% OPTIMZATION 1: 4 Pilots
Np_pilot = 4;

outFile = 'dataset_v36_optimized.h5';
if exist(outFile,'file'); delete(outFile); end

chunkRows = 64;

h5create(outFile, '/X', [numSamples xDim], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) xDim], 'Deflate', 4, 'Shuffle', true);
h5create(outFile, '/Y', [numSamples yDim], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) yDim], 'Deflate', 4, 'Shuffle', true);
h5create(outFile, '/Hrls', [numSamples yDim], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) yDim], 'Deflate', 4, 'Shuffle', true);
h5create(outFile, '/norm_factor', [numSamples 1], ...
    'Datatype','single', 'ChunkSize',[min(numSamples,1024) 1], 'Deflate', 4, 'Shuffle', true);

h5create(outFile, '/meta/Nc', 1, 'Datatype','int32'); h5write(outFile,'/meta/Nc', int32(Nc));
h5create(outFile, '/meta/xDim', 1, 'Datatype','int32'); h5write(outFile,'/meta/xDim', int32(xDim));
% Mark as Optimized
h5create(outFile, '/meta/optimized', 1, 'Datatype','int32'); h5write(outFile,'/meta/optimized', int32(1));

%% Modulators
M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);

pilot_bits = randi([0 1], Nc, 1);
Fpilot = pskmodObj(pilot_bits);

hcom_magnitude = db2mag([0 -9.7 -19.2 -22.8]).';

%% Generate samples
fprintf('=== V3.6 OPTIMIZED: Robust Ridge LS ===\n');
fprintf('Pilot symbols: %d\n', Np_pilot);
fprintf('Regularization: 0.1 (strong noise suppression)\n');
tic;

for idx = 1:numSamples
    if mod(idx, 500) == 0 || idx == 1
        fprintf('Sample %d / %d (%.1f%%)\n', idx, numSamples, 100*idx/numSamples);
    end

    % Radar Tx
    data = randi([0 1], Nc, Ns);
    Ftx_radar = qammod(data, 2, 'UnitAveragePower', true);

    % Random targets
    numTarget = randi([1 5], 1, 1);
    target_power = -25 + 20 * rand(1, numTarget);
    rangeBins = randi([5 floor(r_max/delta_r)], 1, numTarget);
    target_range = rangeBins * delta_r;
    target_velocity = (2*rand(1, numTarget) - 1) * v_max;
    Y_target = generate_target_echo(target_power, target_velocity, target_range, Ftx_radar);

    % Random SI
    numStatic = randi([0 5], 1, 1);
    numSlow = randi([0 15], 1, 1);
    numSI = numStatic + numSlow;
    if numSI > 0
        si_power = -35 + 30 * rand(1, numSI);
        si_rangeBins = randi([1 floor(100/delta_r)], 1, numSI);
        si_range = si_rangeBins * delta_r;
        si_velocity = [zeros(1, numStatic), (2*rand(1, numSlow)-1)*2];
        permIdx = randperm(numSI);
        Y_si = generate_target_echo(si_power(permIdx), si_velocity(permIdx), si_range(permIdx), Ftx_radar);
    else
        Y_si = zeros(Nc, Ns);
    end

    % Noise
    noise_power = randi([0 20], 1, 1);
    R_noise = generate_noise(noise_power);

    % Communication
    Ftx_com = zeros(Nc, Ns);
    for col = 1:Np_pilot
        Ftx_com(:, col) = Fpilot;
    end
    for col = (Np_pilot+1):Ns
        x_com_bit = randi([0 1], 1, log2(M) * Nc);
        Ftx_com(:, col) = pskmodObj(x_com_bit');
    end
    x_com_td = DFT_matrix' * Ftx_com;

    % Com power
    com_power = randi([20 30], 1, 1);
    com_power_sqrt = sqrt(10^(com_power / 10));

    % Random Phase
    random_phase = 2 * pi * rand(4, 1);
    hcom = hcom_magnitude .* exp(1j * random_phase) .* com_power_sqrt;

    % Apply
    x_com_td_vec = reshape(x_com_td, [], 1);
    y_com_td_vec = filter(hcom, 1, x_com_td_vec);
    Y_com_td = reshape(y_com_td_vec, Nc, Ns);
    Y_com = DFT_matrix * Y_com_td;

    % Rx
    A = DFT_matrix * (Y_target + Y_si) + Y_com + R_noise;

    % FDIC
    H_comp_mat = zeros(Nc, Ns);
    E = fdic(A, Ftx_radar, H_comp_mat);

    % OPTIMIZATION 2: Robust LS
    e_td = DFT_matrix' * E;
    x_pilot_td = x_com_td(:, 1:Np_pilot);
    y_pilot_td = e_td(:, 1:Np_pilot);
    % Change lambda to 0.1 for robustness
    h_ls = ls_fir_from_pilot_robust(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length, 0.1);

    % Features
    norm_factor = sqrt(mean(abs(E(:)).^2)) + eps;
    E_n = E ./ norm_factor;
    P_sc = mean(abs(E_n).^2, 2);
    logP = log10(P_sc + 1e-12);

    band_mean = zeros(B,1);
    band_std = zeros(B,1);
    band_max = zeros(B,1);
    for b = 1:B
        s = (b-1)*bandSize + 1;
        t = b*bandSize;
        v = logP(s:t);
        band_mean(b) = mean(v);
        band_std(b) = std(v);
        band_max(b) = max(v);
    end

    g_mean = mean(logP); g_std = std(logP);
    g_max = max(logP); g_min = min(logP);
    g_p = prctile(logP, percentiles);
    g = [g_mean; g_std; g_max; g_min; g_p(:)];

    nf_feat = single(log10(norm_factor));
    hrls8 = single([real(h_ls(:)); imag(h_ls(:))]);

    x_in = single([band_mean; band_std; band_max; g; nf_feat; hrls8]);
    y_out = single([real(hcom); imag(hcom)]);

    h5write(outFile, '/X', x_in.', [idx 1], [1 xDim]);
    h5write(outFile, '/Y', y_out.', [idx 1], [1 yDim]);
    h5write(outFile, '/Hrls', hrls8.', [idx 1], [1 yDim]);
    h5write(outFile, '/norm_factor', single(norm_factor), [idx 1], [1 1]);
end

elapsed = toc;
fprintf('Optimized dataset generated in %.1f s\n', elapsed);

%% Optimization Helper
function h_hat = ls_fir_from_pilot_robust(x, y, L, lambda)
    % Robust LS with Tikhonov Regularization (Ridge)
    x = x(:); y = y(:);
    N = length(x);
    if N <= L, error('Not enough samples'); end
    rows = N - L + 1;
    Xc = zeros(rows, L);
    for kk = 1:rows
        Xc(kk, :) = x(kk+L-1:-1:kk).';
    end
    yv = y(L:end);
    % Ridge Regression: (X'X + lambda*I)^-1 X'y
    h_hat = (Xc' * Xc + lambda * eye(L)) \ (Xc' * yv);
end
