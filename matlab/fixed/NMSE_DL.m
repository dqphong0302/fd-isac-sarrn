clc; clear; close all;
% Test_NMSE_DL_only.m
% NMSE evaluation: GSFDIC + DL (no LS, no RLS)

if ~isdeployed
    addpath('./useful_function/');
end

%% Python + ONNX Setup
pe = pyenv;
insert(py.sys.path, int32(0), pwd);
try
    pymod = py.importlib.import_module('onnx_inference_helper');
    py.importlib.reload(pymod);
catch ME
    fprintf('\n===== PYTHON IMPORT ERROR =====\n');
    disp(ME.message);
    error('Stop execution because Python module failed.');
end

np_py = py.importlib.import_module('numpy');
onnxFile = 'v36_opt.onnx';
if ~isfile(onnxFile)
    error('ONNX file not found: %s', onnxFile);
end

global Nc Ns fC c Delta_f Ts Te;

%% System Parameters
c = 3e8;
Delta_f = 78.125e3;
Nc = 1024;
Ns = 512;
fC = 5e9;
Te = 1/Delta_f;
Ts = Te;
expandVector = @(v, m) v(:, ones(1, m));

n = 0:Nc-1;
k = n';
DFT_matrix = 1/sqrt(Nc) * exp(-2i * pi * k * n / Nc);

bitsPerSymbol = 1;
qam = 2^(bitsPerSymbol);

si_power = [100 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10];
si_velocity = [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
si_range = [1 10 10 10 10 10 11 11 12 12 13 11 11 12 12 13];

target_power = [-10 -10 -10];
target_velocity = [-20 -10 15];
target_range = [5 12 20];
k_target = length(target_power);

M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);
pskdemodObj = comm.PSKDemodulator(M, 'BitOutput', true);

hcom_length = 4;

%% DL Parameters
Np_pilot_dl = 1;
Ridge_Lambda = 0.1;
B = 64;
bandSize = Nc / B;
percentiles = [10 25 50 75 90];
no_of_fb_dl = 2;

%% Sweep SNR
com_power_dB = 0:2:30;
noise_power = 0;
Nmc = 50;
NMSE_dl  = zeros(length(com_power_dB), Nmc);

tic;
fprintf('NMSE evaluation: GSFDIC + DL | Noise=%d dB, Nmc=%d\n\n', noise_power, Nmc);

for pw_idx = 1:length(com_power_dB)
    com_pw = com_power_dB(pw_idx);
    com_power_sqrt = sqrt(10^(com_pw/10));

    for mc = 1:Nmc
        rng(pw_idx * 100 + mc);

        %% Transmit radar
        data = randi([0 qam-1], Nc, Ns);
        Ftx_radar = qammod(data, qam, 'UnitAveragePower', true);

        Y_target = generate_target_echo(target_power, target_velocity, target_range, Ftx_radar);
        Y_si = generate_target_echo(si_power, si_velocity, si_range, Ftx_radar);
        R_noise = generate_noise(noise_power);

        %% Transmit communication
        Ftx_com = zeros(Nc, Ns);
        Mat_x_com_bit = zeros(log2(M)*Nc, Ns);

        for col = 1:Ns
            bits = randi([0 1], log2(M)*Nc, 1);
            Ftx_com(:, col) = pskmodObj(bits);
            Mat_x_com_bit(:, col) = bits;
        end

        x_com_td_mat = DFT_matrix' * Ftx_com;
        x_com_td_vec = reshape(x_com_td_mat, [], 1);
        Ftx_com_vec = reshape(Ftx_com, [], 1);

        hcom_true = db2mag([0 -9.7 -19.2 -22.8]).' ...
            .* exp(1j*[0 -.8 1.6 -2.6].') ...
            .* com_power_sqrt;

        y_com_td = filter(hcom_true, 1, x_com_td_vec);
        Y_com = reshape(y_com_td, Nc, Ns);

        %% Received signal
        A = DFT_matrix * (Y_target + Y_si + Y_com) + R_noise;

        %% ========== GSFDIC + DL ==========
        Y_target_rec = zeros(Nc, Ns);
        Y_com_rec = zeros(Nc, Ns);

        for fb = 1:no_of_fb_dl

            Y_target_rec_prev = Y_target_rec;
            Y_com_rec_prev = Y_com_rec;

            A_tilde = A - Y_com_rec_prev - Y_target_rec_prev;

            if fb > 1
                H_comp = -1/(Ns-1)*((A_tilde(:,1)./Ftx_radar(:,1)) ...
                    - (A_tilde(:,end)./Ftx_radar(:,end)));
                H_comp_mat = expandVector(H_comp, Ns);
            else
                H_comp_mat = zeros(Nc, Ns);
            end

            E_si = fdic(A_tilde, Ftx_radar, H_comp_mat);
            E_radar = E_si + Y_target_rec_prev;

            RDM = calc_periodogram(E_radar, Ftx_radar, 1/80, 1/80);
            targetout = cfar_ofdm_radar(RDM, k_target, 1, 3, 3, 3, 3);

            Y_target_rec = DFT_matrix * ...
                generate_target_echo(target_power, ...
                targetout.velocities, ...
                targetout.ranges, Ftx_radar);

            E_tilde = E_si + Y_target_rec_prev ...
                - Y_target_rec + Y_com_rec_prev;

            %% DL channel estimation
            e_td = DFT_matrix' * E_tilde;
            x_pilot_td = reshape(x_com_td_mat(:, 1:Np_pilot_dl), [], 1);
            y_pilot_td = reshape(e_td(:, 1:Np_pilot_dl), [], 1);

            hcom_est_dl = ls_fir_from_pilot_robust( ...
                x_pilot_td, y_pilot_td, ...
                hcom_length, Ridge_Lambda);

            %% Feature extraction
            norm_factor = sqrt(mean(abs(E_tilde(:)).^2)) + eps;
            E_n = E_tilde ./ norm_factor;

            logP = log10(mean(abs(E_n).^2, 2) + 1e-12);

            bm = arrayfun(@(b) mean(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
            bs = arrayfun(@(b) std(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
            bx = arrayfun(@(b) max(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
            gp = prctile(logP, percentiles);

            gv = [mean(logP); std(logP); max(logP); min(logP); gp(:)];

            xf = single([bm; bs; bx; gv; log10(norm_factor); ...
                single([real(hcom_est_dl(:)); imag(hcom_est_dl(:))])]);

            %% ONNX inference
            yl = pymod.predict_hcom( ...
                np_py.array(xf.', pyargs('dtype', np_py.float32)), ...
                char(onnxFile));

            yh = double(cell2mat(cell(yl)));
            yh = yh(:);

            hcom_est_dl = yh(1:4) + 1j*yh(5:8);

            %% Equalize & reconstruct
            Ec_td = reshape(DFT_matrix' * E_tilde, [], 1);
            Ec = filter(1, hcom_est_dl.', Ec_td);   % FIX: ' is the conjugate transpose -> was equalising with conj(h)
            Ec = DFT_matrix * reshape(Ec, Nc, Ns);

            Ftx_com_est = zeros(Nc, Ns);

            for kk = 1:Ns
                bits_est = pskdemodObj(Ec(:,kk));
                Ftx_com_est(:,kk) = pskmodObj(bits_est);
            end

            Fv = reshape(Ftx_com_est, [], 1);
            Fv(1:Nc*Np_pilot_dl) = Ftx_com_vec(1:Nc*Np_pilot_dl);
            Ftx_com_est = reshape(Fv, Nc, Ns);

            x_est = reshape(DFT_matrix' * Ftx_com_est, [], 1);

            Y_com_rec = DFT_matrix * ...
                reshape(filter(hcom_est_dl, 1, x_est), Nc, Ns);
        end

        %% NMSE
        NMSE_dl(pw_idx, mc) = ...
            norm(hcom_est_dl - hcom_true)^2 / ...
            (norm(hcom_true)^2 + eps);
        fprintf('SNR=%2d dB mc=%d | DL=%.4e\n', ...
            com_pw, mc, NMSE_dl(pw_idx,mc));
    end
end

elapsed = toc;
fprintf('\nTotal time: %.1f s\n', elapsed);

%% Average NMSE
NMSE_dl_mean = smooth(mean(NMSE_dl, 2));

%%
NMSE_dl_mean = NMSE_dl_mean';
%% Plot
figure('Position', [100 100 800 600]);

plot(com_power_dB, NMSE_dl_mean, '-s', ...
    'LineWidth', 1.8, ...
    'MarkerSize', 7);

grid on;
xlabel('$\gamma_c$ (dB)', 'Interpreter', 'latex','FontSize', 16);
ylabel('NMSE', 'FontSize', 14);
% ylim([10e-7 3*10e-4])
title('Channel Estimation NMSE of GSFDIC+DL (no RLS)', 'FontSize', 15);
legend('GSFDIC + DL', 'Location', 'northeast');

if ~isfolder('figures_comparison'), mkdir('figures_comparison'); end
print(gcf, fullfile('figures_comparison', ...
    'nmse_dl_only_noRLS.png'), '-dpng', '-r200');

fprintf('Saved: figures_comparison/nmse_dl_only_noRLS.png\n');

%% Helper
function h = ls_fir_from_pilot_robust(x, y, L, lam)
x = x(:);
y = y(:);
rows = length(x) - L + 1;
Xc = zeros(rows, L);
for k = 1:rows
    Xc(k,:) = x(k+L-1:-1:k).';
end
h = (Xc'*Xc + lam*eye(L)) \ (Xc'*y(L:end));
end