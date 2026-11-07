% isac_with_dl_v36.m
% -------------------------------------------------------------------------
% Full-Duplex ISAC Processing with Deep Learning-based Channel Estimation
% Based on: goc/isac_anh_phong.m
% DL Model: V3.6 (Residual Ridge-MLP, 4 Pilots)
% -------------------------------------------------------------------------
clc; clear; close all;

if ~isdeployed
    addpath('./useful_function/');
end

%% Fix random seed for reproducibility (same results every run)
rng(42);

%% Python + ONNX Setup
pe = pyenv;
try
    pymod = py.importlib.import_module("onnx_inference_helper");
    py.importlib.reload(pymod);
catch ME
    warning("Python module helper issue: %s", ME.message);
end
np = py.importlib.import_module("numpy");
onnxFile = "v36_opt.onnx";
if ~isfile(onnxFile), error('ONNX file not found: %s', onnxFile); end

%% System Parameters (WIFI 6 standard, 5 GHz)
global Nc Ns fC c Delta_f Ts Te;
c = 3e8;
Delta_f = 78.125e3;
Nc = 1024;        % Number of Subcarriers
Ns = 512;         % Number of Symbols
fC = 5e9;
delta_r = 1.1574;
r_max = 2500;
delta_v = 2.68;
v_max = 75;
Te = 1/Delta_f;
Ts = Te;
lambda_rf = c / fC;
expandVector = @(v, m) v(:, ones(1, m));

n = 0:Nc-1;
k = n';
DFT_matrix = 1/sqrt(Nc) * exp(-2i * pi * k * n / Nc);

%% DL V3.6 Parameters
Np_pilot_dl = 4;       % Number of pilot OFDM symbols for DL
Ridge_Lambda = 0.1;    % Ridge Regularization parameter
hcom_length = 4;       % Number of communication chsannel taps
B = 64;                % Number of sub-bands for feature extraction
percentiles = [10 25 50 75 90];

%% SI and Target Configuration
si_power = [100 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10];
si_velocity = [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
si_range = [1 10 10 10 10 10 11 11 12 12 13 11 11 12 12 13];

target_power = [-10 -10 -10];
target_velocity = [12 -17 -15];
target_range = [5 10 15];
k_target = length(target_power);

no_of_fb = 1; % Reduced from 4 - DL V3.6 enables single-pass processing!

%% Modulation Setup
bitsPerSymbol = 1;
qam = 2^(bitsPerSymbol);
data = randi([0 qam - 1], Nc, Ns);
power_radar_dB = 0;
Ftx_radar = qammod(data, qam, 'UnitAveragePower', true);

M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);
pskdemodObj = comm.PSKDemodulator(M, 'BitOutput', true);

% Known pilot symbol for DL estimation
pilot_bits = randi([0 1], Nc, 1);
Fpilot = pskmodObj(pilot_bits);

%% Loop over Communication Power Levels
noise_power = 0; % Environment Noise
com_power_levels = 30; % Can be expanded to 20:2:30
E_output_final = cell(size(com_power_levels));

for com_pw_idx = 1:length(com_power_levels)
    com_power = com_power_levels(com_pw_idx);
    com_power_sqrt = sqrt(10^(com_power/10));
    
    fprintf('Processing Com Power: %d dB\n', com_power);
    
    %% Signal Generation
    Y_target = generate_target_echo(target_power, target_velocity, target_range, Ftx_radar);
    Y_si = generate_target_echo(si_power, si_velocity, si_range, Ftx_radar);
    R_noise = generate_noise(noise_power);
    
    % Communication Signal (ALL random - SAME as original!)
    Ftx_com = zeros(Nc, Ns);
    Mat_x_com_bit = zeros(log2(M) * Nc, Ns);
    for col = 1:Ns
        x_com_bit = randi([0 1], 1, log2(M) * Nc);
        x_com = pskmodObj(x_com_bit');
        Ftx_com(:, col) = x_com;
        Mat_x_com_bit(:, col) = x_com_bit';
    end
    x_com_td = DFT_matrix' * Ftx_com;
    x_com_td_vec = reshape(x_com_td, [], 1);
    Ftx_com_vec = reshape(Ftx_com, [], 1);  % Added - same as original
    
    % True Communication Channel (FIXED phase - SAME as original!)
    % Original: hcom = db2mag([0 -9.7 -19.2 -22.8]).*exp(1j*[0 -.8 1.6 -2.6]).*com_power_sqrt_temp;
    hcom_true = db2mag([0 -9.7 -19.2 -22.8]).' .* exp(1j*[0 -.8 1.6 -2.6].') .* com_power_sqrt;
    
    y_com_td_vec = filter(hcom_true, 1, x_com_td_vec);
    Y_com = reshape(y_com_td_vec, Nc, Ns); % TIME domain (NOT frequency!) - same as original!
    
    % Received Signal at ISAC Node - Signal combination in TIME domain, then DFT
    A = DFT_matrix * (Y_target + Y_si + Y_com) + R_noise;
    
    % ============================================================
    % DL V3.6: Estimate hcom ONCE from raw FDIC output (before loop)
    % This is how the model was trained!
    % ============================================================
    E_dl_raw = fdic(A, Ftx_radar, zeros(Nc, Ns)); % Raw FDIC - no iterative refinement
    e_td_raw = DFT_matrix' * E_dl_raw;
    x_pilot_td = x_com_td(:, 1:Np_pilot_dl);
    y_pilot_td = e_td_raw(:, 1:Np_pilot_dl);
    
    % Ridge Regularized Least Squares (Linear Anchor)
    hcom_est_ridge = ls_fir_from_pilot_robust(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length, Ridge_Lambda);
    
    % Feature Extraction for DL (from raw FDIC output)
    norm_factor = sqrt(mean(abs(E_dl_raw(:)).^2)) + eps;
    E_n = E_dl_raw ./ norm_factor;
    logP = log10(mean(abs(E_n).^2, 2) + 1e-12);
    
    bandSize = Nc / B;
    band_mean = arrayfun(@(b) mean(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
    band_std = arrayfun(@(b) std(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
    band_max = arrayfun(@(b) max(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
    g_p = prctile(logP, percentiles);
    g = [mean(logP); std(logP); max(logP); min(logP); g_p(:)];
    
    x_vec = single([band_mean; band_std; band_max; g; log10(norm_factor); single([real(hcom_est_ridge(:)); imag(hcom_est_ridge(:))])]);
    
    % ONNX Inference via Python (ONE-TIME, before loop)
    y_list = pymod.predict_hcom(np.array(x_vec.', pyargs('dtype', np.float32)), char(onnxFile));
    yhat8 = double(cell2mat(cell(y_list))); yhat8 = yhat8(:);
    hcom_est = yhat8(1:4) + 1j*yhat8(5:8); % DL-estimated complex channel
    
    fprintf('  DL Channel Est. Complete: NMSE = %.2f dB\n', ...
        10*log10(norm(hcom_est - hcom_true)^2 / (norm(hcom_true)^2 + eps)));
    
    %% GSFDIC Processing - 2 Phases with DL-estimated hcom
    Y_target_rec = zeros(Nc, Ns);
    Y_com_rec = zeros(Nc, Ns);
    H_comp_mat = zeros(Nc, Ns);
    
    % ================================================================
    % PHASE 1: Remove most SI and Communication signal
    % ================================================================
    fprintf('  Phase 1: Removing SI and Communication...\n');
    
    % 1.1 Initial FDIC (no H_comp yet)
    A_tilde = A;  % First iteration, no previous estimates
    E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);
    
    % 1.2 Initial target detection
    E_radar_comp = E_si_suppress_tilde;
    RDM = calc_periodogram(E_radar_comp, Ftx_radar, 1/100, 1/100);
    targetout = cfar_ofdm_radar(RDM, k_target, 1, 3, 3, 3, 3);
    Y_target_rec = DFT_matrix * generate_target_echo(target_power, targetout.velocities, targetout.ranges, Ftx_radar);
    
    % 1.3 Communication signal reconstruction (using DL hcom)
    E_tilde = E_si_suppress_tilde - Y_target_rec;
    Ec_td = reshape(DFT_matrix' * E_tilde, [], 1);
    Ec = DFT_matrix * reshape(filter(1, hcom_est', Ec_td), Nc, Ns);
    
    % 1.4 Bits decode and re-modulate
    Ftx_com_est = zeros(Nc, Ns);
    for kk = 1:Ns
        bits_est = pskdemodObj(Ec(:, kk));
        Ftx_com_est(:, kk) = pskmodObj(bits_est);
    end
    Ftx_com_vec_local = reshape(Ftx_com, [], 1);
    Ftx_com_est = reshape(Ftx_com_est, [], 1);
    N_pilot_samples = Np_pilot_dl * Nc;
    Ftx_com_est(1:N_pilot_samples) = Ftx_com_vec_local(1:N_pilot_samples);
    Ftx_com_est = reshape(Ftx_com_est, Nc, Ns);
    
    % 1.5 Reconstruct communication signal
    x_com_est = reshape(DFT_matrix' * Ftx_com_est, [], 1);
    Y_com_rec = DFT_matrix * reshape(filter(hcom_est', 1, x_com_est), Nc, Ns);
    
    % ================================================================
    % PHASE 2: Remove residual + Refine target detection
    % ================================================================
    fprintf('  Phase 2: Refining target detection...\n');
    
    % 2.1 Subtract reconstructed signals
    Y_target_rec_prev = Y_target_rec;
    Y_com_rec_prev = Y_com_rec;
    A_tilde = A - Y_com_rec_prev - Y_target_rec_prev;
    
    % 2.2 Compute H_comp for slow-moving objects
    H_comp = -1/(Ns - 1) * ((A_tilde(:,1) ./ Ftx_radar(:,1)) - (A_tilde(:,end) ./ Ftx_radar(:,end)));
    H_comp_mat = expandVector(H_comp, Ns);
    
    % 2.3 FDIC with compensation
    E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);
    
    % 2.4 Final target detection
    E_radar_comp_gifdic = E_si_suppress_tilde + Y_target_rec_prev;
    RDM = calc_periodogram(E_radar_comp_gifdic, Ftx_radar, 1/100, 1/100);
    targetout = cfar_ofdm_radar(RDM, k_target, 1, 3, 3, 3, 3);
    Y_target_rec = DFT_matrix * generate_target_echo(target_power, targetout.velocities, targetout.ranges, Ftx_radar);
    
    % 2.5 Final communication refinement (optional, for completeness)
    E_tilde = E_si_suppress_tilde + Y_target_rec_prev - Y_target_rec + Y_com_rec_prev;
    Ec_td = reshape(DFT_matrix' * E_tilde, [], 1);
    Ec = DFT_matrix * reshape(filter(1, hcom_est', Ec_td), Nc, Ns);
    
    Ftx_com_est = zeros(Nc, Ns);
    for kk = 1:Ns
        bits_est = pskdemodObj(Ec(:, kk));
        Ftx_com_est(:, kk) = pskmodObj(bits_est);
    end
    Ftx_com_est = reshape(Ftx_com_est, [], 1);
    Ftx_com_est(1:N_pilot_samples) = Ftx_com_vec_local(1:N_pilot_samples);
    Ftx_com_est = reshape(Ftx_com_est, Nc, Ns);
    
    x_com_est = reshape(DFT_matrix' * Ftx_com_est, [], 1);
    Y_com_rec = DFT_matrix * reshape(filter(hcom_est', 1, x_com_est), Nc, Ns);
    
    E_output_final{com_pw_idx} = E_radar_comp_gifdic;
    
    % Final NMSE
    nmse = 10*log10(norm(hcom_est - hcom_true)^2 / (norm(hcom_true)^2 + eps));
    fprintf('Final NMSE (DL V3.6): %.2f dB\n\n', nmse);
end

%% Plotting Range-Doppler Map and Save to File
figFolder = 'figures_comparison';
if ~isfolder(figFolder); mkdir(figFolder); end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

for j = 1:length(com_power_levels)
    fig = figure('Visible', 'on', 'Position', [100 100 700 600]);
    plot_periodogram(E_output_final{j}, Ftx_radar, 1/100, 1/60);
    per_title = sprintf('DL V3.6 ISAC (3 Phases): Pc/Pn = %d dB', com_power_levels(j));
    title(per_title);
    drawnow;  % Force figure to render
    
    % Save to file
    figName = sprintf('DL_V36_Pc%d_%s.png', com_power_levels(j), timestamp);
    print(fig, fullfile(figFolder, figName), '-dpng', '-r150');  % Better quality
    fprintf('Saved: %s\n', fullfile(figFolder, figName));
end

%% Helper Function: Robust LS with Ridge Regularization
function h_hat = ls_fir_from_pilot_robust(x, y, L, lambda)
    x = x(:); y = y(:);
    N = length(x);
    rows = N - L + 1;
    Xc = zeros(rows, L);
    for kk = 1:rows
        Xc(kk, :) = x(kk+L-1:-1:kk).';
    end
    yv = y(L:end);
    h_hat = (Xc' * Xc + lambda * eye(L)) \ (Xc' * yv);
end
