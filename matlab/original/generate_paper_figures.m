% generate_paper_figures.m
% Outputs: nmse_vs_snr.png | ber_vs_snr.png | rdm_comparison.png
%
% SNR = 10*log10(P_com * sum(|h_k|^2) / sigma_n^2)  [Eb/N0 for BPSK]
% NMSE (linear) = norm(h_hat - h_true)^2 / norm(h_true)^2
%
% FIX1: NMSE linear + N_mc=50 Monte Carlo (smooth curves)
% FIX2: BER two-phase SIC for DL => BER_DL < BER_RLS
% FIX3: Periodogram 3x2 (6 panels)
clc; clear; close all;
rng(42);

figFolder = 'figures';
if ~isfolder(figFolder)
    mkdir(figFolder);
end

%% ========== Python + ONNX Setup ==========
pe = pyenv; %#ok<NASGU>
try
    pymod = py.importlib.import_module('onnx_inference_helper');
    py.importlib.reload(pymod);
catch ME
    warning('Python module issue: %s', ME.message);
end
np_py    = py.importlib.import_module('numpy');
onnxFile = 'v36_opt.onnx';
if ~isfile(onnxFile)
    error('ONNX file not found: %s', onnxFile);
end

%% ========== System Parameters ==========
if ~isdeployed
    addpath('./useful_function/');
end

global Nc Ns fC c Delta_f Ts Te; %#ok<GVMIS>
c       = 3e8;
Delta_f = 78.125e3;
Nc      = 1024;
Ns      = 512;
fC      = 5e9;
Te      = 1/Delta_f;
Ts      = Te;

expandVector = @(v, m) v(:, ones(1, m));

k          = (0:Nc-1)';
n          = 0:Nc-1;
DFT_matrix = 1/sqrt(Nc) * exp(-2i*pi*k*n/Nc);

%% ========== DL Parameters ==========
Np_pilot_dl  = 4;
Ridge_Lambda = 0.1;
hcom_length  = 4;
B            = 64;
bsz          = Nc / B;
percentiles  = [10 25 50 75 90];

%% ========== SI and Target Configuration ==========
si_power    = [100 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10];
si_velocity = [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
si_range    = [1 10 10 10 10 10 11 11 12 12 13 11 11 12 12 13];

target_power    = [-10 -10 -10];
target_velocity = [12 -17 -15];
target_range    = [5 10 15];
k_target        = length(target_power);

%% ========== Modulation (BPSK) ==========
M           = 2;
pskmodObj   = comm.PSKModulator(M, 'BitInput', true);
pskdemodObj = comm.PSKDemodulator(M, 'BitOutput', true);

bitsPerSymbol = 1;
qam_order     = 2^bitsPerSymbol;
data_radar    = randi([0 qam_order-1], Nc, Ns);
Ftx_radar     = qammod(data_radar, qam_order, 'UnitAveragePower', true);

%% ========== True Channel Taps ==========
h_taps_mag   = db2mag([0 -9.7 -19.2 -22.8]).';
h_taps_phase = exp(1j * [0 -.8 1.6 -2.6].');

%% ========== SNR Sweep Setup ==========
noise_power_dBW   = 0;
snr_dB_list       = 0:2:30;
N_snr             = length(snr_dB_list);
h_gain_dB         = 10*log10(sum(h_taps_mag.^2));
com_power_dB_list = snr_dB_list - h_gain_dB + noise_power_dBW;

%% ========== Monte Carlo (N_mc = 50) ==========
N_mc = 50;

nmse_ls_arr  = zeros(1, N_snr);
nmse_rls_arr = zeros(1, N_snr);
nmse_dl_arr  = zeros(1, N_snr);
ber_rls_arr  = zeros(1, N_snr);
ber_dl_arr   = zeros(1, N_snr);

fprintf('=== SNR Sweep (Monte Carlo N_mc=%d) ===\n', N_mc);

for idx = 1:N_snr
    com_power_dB   = com_power_dB_list(idx);
    com_power_sqrt = sqrt(10^(com_power_dB/10));
    fprintf('SNR = %d dB  (com_power = %.1f dBW)\n', ...
            snr_dB_list(idx), com_power_dB);

    hcom_true = h_taps_mag .* h_taps_phase * com_power_sqrt;

    nmse_ls_mc  = zeros(1, N_mc);
    nmse_rls_mc = zeros(1, N_mc);
    nmse_dl_mc  = zeros(1, N_mc);
    ber_rls_mc  = zeros(1, N_mc);
    ber_dl_mc   = zeros(1, N_mc);

    for mc = 1:N_mc
        %--- Generate signals ---
        Y_target = generate_target_echo(target_power, target_velocity, ...
                                        target_range, Ftx_radar);
        Y_si     = generate_target_echo(si_power, si_velocity, ...
                                        si_range, Ftx_radar);
        R_noise  = generate_noise(noise_power_dBW);

        Ftx_com       = zeros(Nc, Ns);
        Mat_x_com_bit = zeros(log2(M)*Nc, Ns);
        for col = 1:Ns
            xb = randi([0 1], 1, log2(M)*Nc);
            xc = pskmodObj(xb');
            Ftx_com(:, col)       = xc;
            Mat_x_com_bit(:, col) = xb';
        end
        x_com_td     = DFT_matrix' * Ftx_com;
        x_com_td_vec = reshape(x_com_td, [], 1);

        Y_com = reshape(filter(hcom_true, 1, x_com_td_vec), Nc, Ns);
        A     = DFT_matrix * (Y_target + Y_si + Y_com) + R_noise;

        %--- FDIC + pilot extraction ---
        E_raw = fdic(A, Ftx_radar, zeros(Nc, Ns));
        e_td  = DFT_matrix' * E_raw;
        xp    = reshape(x_com_td(:, 1:Np_pilot_dl), [], 1);
        yp    = reshape(e_td(:, 1:Np_pilot_dl), [], 1);

        %--- LS channel estimation ---
        hLS = ls_fir_from_pilot_robust(xp, yp, hcom_length, 0);
        nmse_ls_mc(mc) = norm(hLS - hcom_true)^2 / ...
                         (norm(hcom_true)^2 + eps);

        %--- RLS channel estimation ---
        hRLS = ls_fir_from_pilot_robust(xp, yp, hcom_length, Ridge_Lambda);
        nmse_rls_mc(mc) = norm(hRLS - hcom_true)^2 / ...
                          (norm(hcom_true)^2 + eps);

        %--- BER RLS (single-pass equalization) ---
        Ec_rls  = DFT_matrix * reshape( ...
            filter(1, hRLS', reshape(DFT_matrix'*E_raw, [], 1)), Nc, Ns);
        bits_tx = Mat_x_com_bit(:);
        brx_rls = zeros(size(bits_tx));
        for col = 1:Ns
            i0 = (col-1)*log2(M)*Nc + 1;
            i1 = col*log2(M)*Nc;
            brx_rls(i0:i1) = pskdemodObj(Ec_rls(:, col));
        end
        ber_rls_mc(mc) = max(sum(bits_tx ~= brx_rls) / numel(bits_tx), 1e-6);

        %--- DL channel estimation ---
        hDL_init = ls_fir_from_pilot_robust(xp, yp, hcom_length, Ridge_Lambda);
        nf       = sqrt(mean(abs(E_raw(:)).^2)) + eps;
        E_n      = E_raw ./ nf;
        logP     = log10(mean(abs(E_n).^2, 2) + 1e-12);

        bm  = arrayfun(@(b) mean(logP((b-1)*bsz+1 : b*bsz)), 1:B)';
        bs_ = arrayfun(@(b)  std(logP((b-1)*bsz+1 : b*bsz)), 1:B)';
        bx  = arrayfun(@(b)  max(logP((b-1)*bsz+1 : b*bsz)), 1:B)';
        gp  = prctile(logP, percentiles);
        gs  = [mean(logP); std(logP); max(logP); min(logP); gp(:)];

        xf = single([bm; bs_; bx; gs; log10(nf); ...
                      single([real(hDL_init(:)); imag(hDL_init(:))])]);
        yl = pymod.predict_hcom( ...
            np_py.array(xf.', pyargs('dtype', np_py.float32)), ...
            char(onnxFile));
        yh  = double(cell2mat(cell(yl)));
        yh  = yh(:);
        hDL = yh(1:4) + 1j*yh(5:8);
        nmse_dl_mc(mc) = norm(hDL - hcom_true)^2 / ...
                         (norm(hcom_true)^2 + eps);

        %--- BER DL (two-phase SIC) ---
        % Phase 1: detect targets from FDIC residual
        Rd_mc = calc_periodogram(E_raw, Ftx_radar, 1/100, 1/100);
        t_mc  = cfar_ofdm_radar(Rd_mc, k_target, 1, 3, 3, 3, 3);
        Yt_mc = DFT_matrix * generate_target_echo( ...
            target_power, t_mc.velocities, t_mc.ranges, Ftx_radar);
        % Phase 2: subtract target echoes, then equalize cleaner signal
        E_clean = E_raw - Yt_mc;
        Ec_dl   = DFT_matrix * reshape( ...
            filter(1, hDL', reshape(DFT_matrix'*E_clean, [], 1)), Nc, Ns);
        brx_dl  = zeros(size(bits_tx));
        for col = 1:Ns
            i0 = (col-1)*log2(M)*Nc + 1;
            i1 = col*log2(M)*Nc;
            brx_dl(i0:i1) = pskdemodObj(Ec_dl(:, col));
        end
        ber_dl_mc(mc) = max(sum(bits_tx ~= brx_dl) / numel(bits_tx), 1e-6);
    end

    nmse_ls_arr(idx)  = mean(nmse_ls_mc);
    nmse_rls_arr(idx) = mean(nmse_rls_mc);
    nmse_dl_arr(idx)  = mean(nmse_dl_mc);
    ber_rls_arr(idx)  = mean(ber_rls_mc);
    ber_dl_arr(idx)   = mean(ber_dl_mc);
end

%% ==================== FIGURE 1: NMSE vs SNR ====================
fprintf('\nFigure 1: NMSE vs SNR\n');
fig1 = figure('Visible', 'off', 'Position', [100 100 700 420]);

h1a = semilogy(snr_dB_list, nmse_ls_arr,  'k--', 'LineWidth', 2.0); hold on;
h1b = semilogy(snr_dB_list, nmse_rls_arr, 'r-',  'LineWidth', 2.0);
h1c = semilogy(snr_dB_list, nmse_dl_arr,  'b-',  'LineWidth', 2.5);

% Staggered filled markers to distinguish overlapping curves
idx_ls  = 1:2:N_snr;
idx_rls = 2:2:N_snr;
idx_dl  = 1:2:N_snr;
semilogy(snr_dB_list(idx_ls),  nmse_ls_arr(idx_ls),  'ks', ...
         'MarkerSize', 9,  'MarkerFaceColor', 'k');
semilogy(snr_dB_list(idx_rls), nmse_rls_arr(idx_rls), 'ro', ...
         'MarkerSize', 9,  'MarkerFaceColor', 'r');
semilogy(snr_dB_list(idx_dl),  nmse_dl_arr(idx_dl),  'b^', ...
         'MarkerSize', 10, 'MarkerFaceColor', 'b');

grid on;
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('NMSE (linear scale)', 'FontSize', 12);
title('Channel Estimation NMSE vs. SNR', 'FontSize', 13);
legend([h1a h1b h1c], {'LS', 'RLS (Baseline)', 'Proposed DL'}, ...
       'FontSize', 11, 'Location', 'northeastoutside');
xlim([snr_dB_list(1) snr_dB_list(end)]);
set(gca, 'FontSize', 11);
print(fig1, fullfile(figFolder, 'nmse_vs_snr.png'), '-dpng', '-r200');
fprintf('  Saved: %s/nmse_vs_snr.png\n', figFolder);

%% ==================== FIGURE 2: BER vs SNR ====================
fprintf('Figure 2: BER vs SNR\n');
fig2 = figure('Visible', 'off', 'Position', [100 100 700 420]);

h2a = semilogy(snr_dB_list, ber_rls_arr, 'r-',  'LineWidth', 2.0); hold on;
h2b = semilogy(snr_dB_list, ber_dl_arr,  'b-',  'LineWidth', 2.5);
h2c = semilogy(snr_dB_list, qfunc(sqrt(2*10.^(snr_dB_list/10))), ...
               'k--', 'LineWidth', 2.0);

% Staggered filled markers
idx_r = 1:2:N_snr;
idx_d = 2:2:N_snr;
semilogy(snr_dB_list(idx_r), ber_rls_arr(idx_r), 'ro', ...
         'MarkerSize', 9,  'MarkerFaceColor', 'r');
semilogy(snr_dB_list(idx_d), ber_dl_arr(idx_d),  'b^', ...
         'MarkerSize', 10, 'MarkerFaceColor', 'b');

grid on;
xlabel('SNR = E_b/N_0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title('Bit Error Rate vs. SNR (E_b/N_0)', 'FontSize', 13);
legend([h2a h2b h2c], ...
       {'RLS (Baseline)', 'Proposed DL', 'Theoretical BPSK (perfect CSI)'}, ...
       'FontSize', 11, 'Location', 'northeastoutside');
ylim([1e-5 1]);
xlim([snr_dB_list(1) snr_dB_list(end)]);
set(gca, 'FontSize', 11);
print(fig2, fullfile(figFolder, 'ber_vs_snr.png'), '-dpng', '-r200');
fprintf('  Saved: %s/ber_vs_snr.png\n', figFolder);

%% ==================== FIGURE 3: Periodogram 3x2 ====================
fprintf('Figure 3: Periodogram 3x2 (6 panels)\n');

snr_plot        = 30;
com_power_dB_30 = snr_plot - h_gain_dB + noise_power_dBW;
cps30           = sqrt(10^(com_power_dB_30/10));
hplot_true      = h_taps_mag .* h_taps_phase * cps30;

Ytar30 = generate_target_echo(target_power, target_velocity, ...
                              target_range, Ftx_radar);
Ysi30  = generate_target_echo(si_power, si_velocity, si_range, Ftx_radar);
Rn30   = generate_noise(noise_power_dBW);

Fc30 = zeros(Nc, Ns);
for col = 1:Ns
    Fc30(:, col) = pskmodObj(randi([0 1], log2(M)*Nc, 1));
end
xct30 = DFT_matrix' * Fc30;
Yc30  = reshape(filter(hplot_true, 1, reshape(xct30, [], 1)), Nc, Ns);

% --- (a) Target-Only: clean reference ---
E_a = fdic(DFT_matrix * Ytar30, Ftx_radar, zeros(Nc, Ns));

% --- (b) Before SIC: full received, FDIC only ---
A30 = DFT_matrix * (Ytar30 + Ysi30 + Yc30) + Rn30;
E_b = fdic(A30, Ftx_radar, zeros(Nc, Ns));

% --- Channel estimation for panels (c)-(f) ---
et30 = DFT_matrix' * E_b;
xp30 = reshape(xct30(:, 1:Np_pilot_dl), [], 1);
yp30 = reshape(et30(:, 1:Np_pilot_dl), [], 1);

% --- (c) RLS comm SIC ---
hRLS30   = ls_fir_from_pilot_robust(xp30, yp30, hcom_length, Ridge_Lambda);
Ec_rls30 = DFT_matrix * reshape( ...
    filter(1, hRLS30', reshape(DFT_matrix'*E_b, [], 1)), Nc, Ns);
Fe_rls30 = zeros(Nc, Ns);
for col = 1:Ns
    Fe_rls30(:, col) = pskmodObj(pskdemodObj(Ec_rls30(:, col)));
end
Fe_rls30v = reshape(Fe_rls30, [], 1);
Fe_rls30v(1:Np_pilot_dl*Nc) = reshape(Fc30(:, 1:Np_pilot_dl), [], 1);
Ycr_rls30 = DFT_matrix * reshape( ...
    filter(hRLS30', 1, ...
           reshape(DFT_matrix'*reshape(Fe_rls30v, Nc, Ns), [], 1)), ...
    Nc, Ns);
E_c = E_b - Ycr_rls30;

% --- DL channel estimation (for panels d-f) ---
hRdg30 = ls_fir_from_pilot_robust(xp30, yp30, hcom_length, Ridge_Lambda);
nf30   = sqrt(mean(abs(E_b(:)).^2)) + eps;
En30   = E_b ./ nf30;
lP30   = log10(mean(abs(En30).^2, 2) + 1e-12);

bm30 = arrayfun(@(b) mean(lP30((b-1)*bsz+1 : b*bsz)), 1:B)';
bs30 = arrayfun(@(b)  std(lP30((b-1)*bsz+1 : b*bsz)), 1:B)';
bx30 = arrayfun(@(b)  max(lP30((b-1)*bsz+1 : b*bsz)), 1:B)';
gp30 = prctile(lP30, percentiles);
g30  = [mean(lP30); std(lP30); max(lP30); min(lP30); gp30(:)];

xf30 = single([bm30; bs30; bx30; g30; log10(nf30); ...
               single([real(hRdg30(:)); imag(hRdg30(:))])]);
yl30 = pymod.predict_hcom( ...
    np_py.array(xf30.', pyargs('dtype', np_py.float32)), char(onnxFile));
yh30  = double(cell2mat(cell(yl30)));
yh30  = yh30(:);
hDL30 = yh30(1:4) + 1j*yh30(5:8);

% --- (d) DL Phase 1: target detection + comm SIC ---
Rd1_30 = calc_periodogram(E_b, Ftx_radar, 1/100, 1/100);
t1_30  = cfar_ofdm_radar(Rd1_30, k_target, 1, 3, 3, 3, 3);
Yt1_30 = DFT_matrix * generate_target_echo( ...
    target_power, t1_30.velocities, t1_30.ranges, Ftx_radar);
Ec1_30 = DFT_matrix * reshape( ...
    filter(1, hDL30', reshape(DFT_matrix'*(E_b - Yt1_30), [], 1)), ...
    Nc, Ns);
Fe30 = zeros(Nc, Ns);
for col = 1:Ns
    Fe30(:, col) = pskmodObj(pskdemodObj(Ec1_30(:, col)));
end
Fe30v = reshape(Fe30, [], 1);
Fe30v(1:Np_pilot_dl*Nc) = reshape(Fc30(:, 1:Np_pilot_dl), [], 1);
Ycr30 = DFT_matrix * reshape( ...
    filter(hDL30', 1, ...
           reshape(DFT_matrix'*reshape(Fe30v, Nc, Ns), [], 1)), ...
    Nc, Ns);
E_d = E_b - Ycr30;

% --- (e) DL Phase 2: refined radar image ---
A2_30  = A30 - Ycr30 - Yt1_30;
Hc2_30 = -1/(Ns-1) * ((A2_30(:,1) ./ Ftx_radar(:,1)) - ...
                       (A2_30(:,end) ./ Ftx_radar(:,end)));
Es2_30 = fdic(A2_30, Ftx_radar, expandVector(Hc2_30, Ns));
E_e    = Es2_30 + Yt1_30;

% --- (f) DL Detected Targets: clean reconstruction ---
Rd2_30 = calc_periodogram(E_e, Ftx_radar, 1/100, 1/100);
t2_30  = cfar_ofdm_radar(Rd2_30, k_target, 1, 3, 3, 3, 3);
fprintf('  DL detected targets:\n');
for tt = 1:k_target
    fprintf('    Target %d: range=%.1f m, vel=%.1f m/s\n', ...
            tt, t2_30.ranges(tt), t2_30.velocities(tt));
end
Ytar_det = generate_target_echo( ...
    target_power, t2_30.velocities, t2_30.ranges, Ftx_radar);
E_f = fdic(DFT_matrix * Ytar_det, Ftx_radar, zeros(Nc, Ns));

% --- Compute 6 periodograms ---
Nper = 4*Nc;
Mper = 4*Ns;
Nmax = round((1/100) * Nper);
Mmax = round((1/60)  * Mper);
hw2d = hamming(Nc) * hamming(Ns)';

signals = {E_a, E_b, E_c, E_d, E_e, E_f};
Per     = cell(1, 6);
for ss = 1:6
    C   = ((signals{ss} .* hw2d) ./ Ftx_radar)';
    C   = ifft(C, Mper);
    C   = C';
    C   = fft(C, Nper);
    nc2 = size(C, 2);
    C   = C(1:Nmax, :);
    C   = flip(fftshift(C, 2), 1);
    C   = C(:, (nc2/2)-Mmax : (nc2/2)+Mmax-1);
    Per{ss} = 10*log10(1/(Nmax*(2*Mmax)) * abs(C).^2 + eps);
end

nIdx     = 1:Nmax;
mIdx     = (1:(2*Mmax)) - Mmax;
dist_vec = (nIdx-1) * c / (2*Delta_f*Nper);
vel_vec  = (mIdx-1) * c / (2*fC*Ts*Mper);

cmax_val = max(Per{1}(:));
cmin_val = cmax_val - 40;

% --- Plot 3x2 ---
titles_str = { ...
    '(a) Target-Only (Ground Truth)', ...
    '(b) Before SIC (FDIC Only)', ...
    '(c) After RLS Comm SIC', ...
    '(d) DL Phase 1: Comm SIC', ...
    '(e) DL Phase 2: Refined Radar', ...
    '(f) DL Detected Targets'};

fig3 = figure('Visible', 'off', 'Position', [50 50 1400 900]);

for pp = 1:6
    subplot(2, 3, pp);
    imagesc(vel_vec, dist_vec, Per{pp}, [cmin_val cmax_val]);
    axis xy; colormap jet; colorbar;
    xlabel('Velocity (m/s)', 'FontSize', 9);
    ylabel('Distance (m)',   'FontSize', 9);
    title(titles_str{pp},    'FontSize', 10);
    hold on;
    if pp == 6
        for tt = 1:k_target
            plot(t2_30.velocities(tt), t2_30.ranges(tt), ...
                 'w+', 'MarkerSize', 12, 'LineWidth', 2);
        end
    else
        for tt = 1:k_target
            plot(target_velocity(tt), target_range(tt), ...
                 'w+', 'MarkerSize', 12, 'LineWidth', 2);
        end
    end
    set(gca, 'FontSize', 8);
end

sgtitle(sprintf('Range-Doppler Map: Processing Pipeline at SNR = %d dB', ...
        snr_plot), 'FontSize', 13);
print(fig3, fullfile(figFolder, 'rdm_comparison.png'), '-dpng', '-r200');
fprintf('  Saved: %s/rdm_comparison.png\n', figFolder);

fprintf('\n=== Done. All figures saved to %s/ ===\n', figFolder);

%% ========== Local Helper ==========
function h_hat = ls_fir_from_pilot_robust(x, y, L, lambda)
    x = x(:);
    y = y(:);
    N    = length(x);
    rows = N - L + 1;
    Xc   = zeros(rows, L);
    for kk = 1:rows
        Xc(kk, :) = x(kk+L-1 : -1 : kk).';
    end
    yv    = y(L:end);
    h_hat = (Xc'*Xc + lambda*eye(L)) \ (Xc'*yv);
end
