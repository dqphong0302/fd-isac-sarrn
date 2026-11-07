clc, clear, close all
%% WIFI 6 standard, subcarrier spacing 78.125 kHz, 5 GHz

if ~isdeployed
    addpath('./useful_function/');
end
global Nc Ns fC c Delta_f Ts Te;

%% This code is to implement the Calibrated FDIC for three targets and 2 SI sources case - Figure 3
%% SI and target range and velocity

% Parameters
c = 3e8;
Delta_f = 78.125e3;
Nc = 1024;
Ns = 512;
fC = 5e9;
delta_r = 1.1574;
r_max = 2500;
delta_v = 2.68;
v_max = 75;
Te = 1/Delta_f;
Ts = Te;
lambda = c / fC;
expandVector = @(v, m) v(:, ones(1, m));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bitsPerSymbol = 1;
qam = 2^(bitsPerSymbol);
data = randi([0 qam - 1], Nc, Ns);
Ftx_radar = qammod(data, qam, 'UnitAveragePower', true);

noise_power = 0;

si_power = [100 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10];
si_velocity = [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
si_range = [1 10 10 10 10 10 11 11 12 12 13 11 11 12 12 13];

target_power = [-10 -10 -10];
target_velocity = [12 -17 -15];
target_range = [5 10 15];
n = 0:Nc-1;
k = n';
DFT_matrix = 1/sqrt(Nc) * exp(-2i * pi * k * n / Nc);

no_of_fb = 2;
k_target = length(target_power);

%%%%%%%%%%%%%%%%%%%%%% WITH COMMUNICATION INTERFERENCE %%%%%%%%%%%%%%%%%%%%
M = 2;
pskmod = comm.PSKModulator(M, 'BitInput',true);
pskdemod = comm.PSKDemodulator(M, 'BitOutput',true);

hcom_length = 4;
hcom_est = zeros(hcom_length,1);
epsilon = 0.0001;
P1 = (epsilon^-1)*eye(hcom_length);

N_pilot = 2*Nc;

com_power_dB = 0:2:30;
com_power_sqrt = sqrt(((10.^(com_power_dB/10))));
numP = length(com_power_dB);
NMSE_linear = zeros(numP,1);
MC = 5;              % Monte Carlo runs

%% Target signal power in dB
for com_pw_sqrt_count = 1:numP
    nmse_mc = zeros(MC,1);
    for mc = 1:MC
        %% Transmitted and Received Signals
        % Target & SI Signal
        Y_target = generate_target_echo(target_power, target_velocity, target_range, Ftx_radar);
        Y_si = generate_target_echo(si_power, si_velocity, si_range, Ftx_radar);
        R_noise = generate_noise(noise_power);

        % Communication Signal
        Ftx_com = zeros(Nc, Ns);
        Mat_x_com_bit = zeros(log2(M) * Nc, Ns);
        for col = 1:Ns
            x_com_bit = randi([0 1], 1, log2(M) * Nc);
            x_com = x_com_bit;
            x_com = pskmod(x_com');
            Ftx_com(:, col) = x_com;
            Mat_x_com_bit(:,col) = x_com_bit';
        end
        x_com_td = DFT_matrix'*Ftx_com;
        x_com_td = reshape(x_com_td,[],1);
        Ftx_com_vec = reshape(Ftx_com,[],1);

        com_power_sqrt_temp = com_power_sqrt(com_pw_sqrt_count);
        hcom = db2mag([0 -9.7 -19.2 -22.8]).*exp(1j*[0 -.8 1.6 -2.6]).*com_power_sqrt_temp; % channel com
        h_gain = hcom;
        y_com_td = filter(hcom, 1, x_com_td);
        Y_com = reshape(y_com_td,[],Ns);

        % Received Signal
        A =  DFT_matrix*(Y_target + Y_si + Y_com) + R_noise;
        A_td = reshape(DFT_matrix'*A,[],1);

        %% GSFDIC Processing
        hcom_est = zeros(hcom_length,1);
        P1 = (epsilon^-1)*eye(hcom_length);   % FIX: reset RLS inverse-correlation matrix every frame (was initialised once, line 57)
        Ftx_com_est = zeros(Nc,Ns);
        Y_target_rec = zeros(Nc,Ns);
        Y_com_rec = zeros(Nc,Ns);
        H_comp_mat = zeros(Nc,Ns);
        for fb = 1:no_of_fb
            % Step 1: Target and Communication Removal
            Y_target_rec_prev = Y_target_rec;
            Y_com_rec_prev = Y_com_rec;
            H_comp_mat_prev = H_comp_mat;
            A_tilde = A - Y_com_rec_prev - Y_target_rec_prev;

            if fb > 1
                H_comp = - 1/(Ns - 1) * ((A_tilde(:,1) ./ Ftx_radar(:,1)) - (A_tilde(:,end) ./ Ftx_radar(:,end)));
                H_comp_mat = expandVector(H_comp, Ns);
            else
                H_comp_mat = zeros(Nc, Ns);
            end
            % Step 2: SI Suppression with Generalized FDIC
            E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);


            % Step 3: Target Compensation and Estimation
            E_radar_comp_gifdic = E_si_suppress_tilde + Y_target_rec_prev;
            RDM = calc_periodogram(E_radar_comp_gifdic, Ftx_radar, 1/100, 1/100);
            targetout = cfar_ofdm_radar(RDM,k_target,1,3,3,3,3);
            target_rec_range = targetout.ranges;
            target_rec_velocity = targetout.velocities;
            Y_target_rec = generate_target_echo(target_power,target_rec_velocity,target_rec_range,Ftx_radar);
            Y_target_rec = DFT_matrix * Y_target_rec;


            E_tilde = E_si_suppress_tilde + Y_target_rec_prev - Y_target_rec + Y_com_rec_prev;

            % Step 4: Communication Compensation and Estimation

            if N_pilot > 0
                x_com_pilot = x_com_td(1:N_pilot);
                E_td = DFT_matrix'*E_tilde;
                E_td = reshape(E_td,[],1);

                Ep_td = E_td(1:N_pilot);
                for m = hcom_length:length(x_com_pilot)

                    [hcom_est,P1] = RLS_function(hcom_est,Ep_td(m),x_com_pilot(m:-1:m-(hcom_length -1)),P1);

                end

            else
                E_td = DFT_matrix'*E_tilde;
                E_td = reshape(E_td,[],1);
                for m = hcom_length:length(x_com_td)
                    [hcom_est,P1] = RLS_function(hcom_est,E_td(m),x_com_td(m:-1:m-(hcom_length -1)),P1);
                end
            end
            Ec_td = DFT_matrix' * E_tilde;
            Ec_td = reshape(Ec_td,[],1);
            h_predict = hcom_est';
            Ec = filter(1,h_predict,Ec_td);
            Ec = DFT_matrix * reshape(Ec,[],Ns);

            for kk = 1:Ns     % Bits Decoding
                Mat_x_com_bit_est(:,kk) = pskdemod(Ec(:,kk));
                x_com_est_vec = Mat_x_com_bit_est(:,kk);
                x_com_est_vec = pskmod(x_com_est_vec);
                Ftx_com_est(:, kk) = x_com_est_vec;
            end
            Ftx_com_est = reshape(Ftx_com_est,[],1);
            Ftx_com_est(1:N_pilot) = Ftx_com_vec(1:N_pilot);
            Ftx_com_est = reshape(Ftx_com_est,[],Ns);

            x_com_est = DFT_matrix'*Ftx_com_est;
            x_com_est = reshape(x_com_est,[],1);
            
            Y_com_rec = filter(h_predict,1,x_com_est); % Reconstruct Communication Signal
            Y_com_rec = reshape(Y_com_rec,[],Ns);
            Y_com_rec = DFT_matrix * Y_com_rec;

            % % ===== NMSE COMPUTATION =====
            % h_true = h_gain(1:hcom_length);   % True channel
            % nmse_mc(mc) = norm(h_predict - h_true)^2 / norm(h_true)^2;

        end
       nmse_mc(mc) = norm(h_predict(:) - hcom(:))^2 / (norm(hcom(:))^2 + eps);
    end
        NMSE_linear(com_pw_sqrt_count) = mean(nmse_mc);

  fprintf('Pcom = %3d dB | NMSE = %.4e (linear)\n', ...
        com_power_dB(com_pw_sqrt_count), ...
        NMSE_linear(com_pw_sqrt_count));
end
%%
NMSE_linear = smooth(NMSE_linear);
%%
figure;
plot(com_power_dB, NMSE_linear, '-o', ...
    'LineWidth', 2, 'MarkerSize', 7);
grid on;
xlabel('$\gamma_c$ (dB)', 'Interpreter', 'latex','FontSize', 16);
ylabel('NMSE');
title('NMSE of RLS Channel Estimation');
set(gca,'FontSize',12);

%%
NMSE_linear = NMSE_linear';
toc

% clc, clear, close all
% %% WIFI 6 standard, subcarrier spacing 78.125 kHz, 5 GHz
% 
% if ~isdeployed
%     addpath('./useful_function/');    
% end
% global Nc Ns fC c Delta_f Ts Te;
% %% This code is to implement the Calibrated FDIC for three targets and 2 SI sources case - Figure 3
% %% SI and target range and velocity
% 
% % Parameters
% c = 3e8;
% Delta_f = 78.125e3;
% Nc = 1024;
% Ns = 512;
% fC = 5e9;
% delta_r = 1.1574;
% r_max = 2500;
% delta_v = 2.68;
% v_max = 75;
% Te = 1/Delta_f;
% Ts = Te;
% lambda = c / fC;
% expandVector = @(v, m) v(:, ones(1, m));
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bitsPerSymbol = 1;
% qam = 2^(bitsPerSymbol);
% data = randi([0 qam - 1], Nc, Ns);
% Ftx_radar = qammod(data, qam, 'UnitAveragePower', true);
% 
% noise_power = 0;
% 
% si_power = [100 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10];
% si_velocity = [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
% si_range = [1 10 10 10 10 10 11 11 12 12 13 11 11 12 12 13];
% 
% target_power = [-10 -10 -10];
% target_velocity = [12 -17 -15];
% target_range = [5 10 15];
% n = 0:Nc-1;
% k = n';
% DFT_matrix = 1/sqrt(Nc) * exp(-2i * pi * k * n / Nc);
% 
% no_of_fb = 2;
% k_target = length(target_power);
% 
% %%%%%%%%%%%%%%%%%%%%%% WITH COMMUNICATION INTERFERENCE %%%%%%%%%%%%%%%%%%%%
% M = 2;
% pskmod = comm.PSKModulator(M, 'BitInput',true);
% pskdemod = comm.PSKDemodulator(M, 'BitOutput',true);
% 
% hcom_length = 4;
% hcom_est = zeros(hcom_length,1);
% epsilon = 0.0001;
% P1 = (epsilon^-1)*eye(hcom_length);
% 
% 
% N_pilot = 2*Nc;
% 
% 
% com_power = 30;
% com_power_sqrt = sqrt(((10.^(com_power/10))));
% E_output_gifdic = cell(size(com_power));
% E_output_fdic = cell(size(com_power));
% 
% 
% qam = 2^(bitsPerSymbol);
% data = randi([0 qam - 1], Nc, Ns);
% R_noise = generate_noise(noise_power);
% 
% %% Target signal power in dB
% for com_pw_sqrt_count = 1:length(com_power)
%     %% Transmitted and Received Signals
%     % Target & SI Signal    
%     Y_target = generate_target_echo(target_power, target_velocity, target_range, Ftx_radar);
%     Y_si = generate_target_echo(si_power, si_velocity, si_range, Ftx_radar);
%     R_noise = generate_noise(noise_power);
% 
%     % Communication Signal
%     Ftx_com = zeros(Nc, Ns);
%     Mat_x_com_bit = zeros(log2(M) * Nc, Ns);
%     for col = 1:Ns
%         x_com_bit = randi([0 1], 1, log2(M) * Nc);
%         x_com = x_com_bit;
%         x_com = pskmod(x_com');
%         Ftx_com(:, col) = x_com;
%         Mat_x_com_bit(:,col) = x_com_bit';
%     end
%     x_com_td = DFT_matrix'*Ftx_com;
%     x_com_td = reshape(x_com_td,[],1);
%     Ftx_com_vec = reshape(Ftx_com,[],1);
% 
%     com_power_sqrt_temp = com_power_sqrt(com_pw_sqrt_count);
%     hcom = db2mag([0 -9.7 -19.2 -22.8]).*exp(1j*[0 -.8 1.6 -2.6]).*com_power_sqrt_temp; % channel com
%     y_com_td = filter(hcom, 1, x_com_td);
%     Y_com = reshape(y_com_td,[],Ns);
% 
%     % Received Signal
%     A =  DFT_matrix*(Y_target + Y_si + Y_com) + R_noise;  
%     A_td = reshape(DFT_matrix'*A,[],1);
% 
%     %% GSFDIC Processing
%     hcom_est = zeros(hcom_length,1);
%     Ftx_com_est = zeros(Nc,Ns);
%     Y_target_rec = zeros(Nc,Ns);
%     Y_com_rec = zeros(Nc,Ns);
%     H_comp_mat = zeros(Nc,Ns);
%     for fb = 1:no_of_fb
%         % Step 1: Target and Communication Removal
%         Y_target_rec_prev = Y_target_rec;
%         Y_com_rec_prev = Y_com_rec;
%         H_comp_mat_prev = H_comp_mat;
%         A_tilde = A - Y_com_rec_prev - Y_target_rec_prev;        
% 
%         if fb > 1
%             H_comp = - 1/(Ns - 1) * ((A_tilde(:,1) ./ Ftx_radar(:,1)) - (A_tilde(:,end) ./ Ftx_radar(:,end)));
%             H_comp_mat = expandVector(H_comp, Ns);
%         else
%             H_comp_mat = zeros(Nc, Ns);
%         end
%         % Step 2: SI Suppression with Generalized FDIC
%         E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);
% 
% 
%         % Step 3: Target Compensation and Estimation
%         E_radar_comp_gifdic = E_si_suppress_tilde + Y_target_rec_prev;
%         RDM = calc_periodogram(E_radar_comp_gifdic, Ftx_radar, 1/100, 1/100);
%         targetout = cfar_ofdm_radar(RDM,k_target,1,3,3,3,3);
%         target_rec_range = targetout.ranges;
%         target_rec_velocity = targetout.velocities;
%         Y_target_rec = generate_target_echo(target_power,target_rec_velocity,target_rec_range,Ftx_radar);
%         Y_target_rec = DFT_matrix * Y_target_rec;
% 
% 
%         E_tilde = E_si_suppress_tilde + Y_target_rec_prev - Y_target_rec + Y_com_rec_prev;
% 
%         % Step 4: Communication Compensation and Estimation
% 
%         if N_pilot > 0
%             x_com_pilot = x_com_td(1:N_pilot);
%             E_td = DFT_matrix'*E_tilde;
%             E_td = reshape(E_td,[],1);
% 
%             Ep_td = E_td(1:N_pilot);
%             for m = hcom_length:length(x_com_pilot)
% 
%                 [hcom_est,P1] = RLS_function(hcom_est,Ep_td(m),x_com_pilot(m:-1:m-(hcom_length -1)),P1);
% 
%             end
% 
%         else
%             E_td = DFT_matrix'*E_tilde;
%             E_td = reshape(E_td,[],1);
%             for m = hcom_length:length(x_com_td)
% 
%                 [hcom_est,P1] = RLS_function(hcom_est,E_td(m),x_com_td(m:-1:m-(hcom_length -1)),P1);
% 
%             end
% 
% 
%         end
% 
%         Ec_td = DFT_matrix' * E_tilde;
%         Ec_td = reshape(Ec_td,[],1);
%         Ec = filter(1,hcom_est',Ec_td);
%         Ec = DFT_matrix * reshape(Ec,[],Ns);
% 
%         for kk = 1:Ns     % Bits Decoding
%             Mat_x_com_bit_est(:,kk) = pskdemod(Ec(:,kk));
%             x_com_est_vec = Mat_x_com_bit_est(:,kk);
%             x_com_est_vec = pskmod(x_com_est_vec);
%             Ftx_com_est(:, kk) = x_com_est_vec;
%         end
%         Ftx_com_est = reshape(Ftx_com_est,[],1);
%         Ftx_com_est(1:N_pilot) = Ftx_com_vec(1:N_pilot);
%         Ftx_com_est = reshape(Ftx_com_est,[],Ns);
% 
%         x_com_est = DFT_matrix'*Ftx_com_est;
%         x_com_est = reshape(x_com_est,[],1);
%         h_predict = hcom_est';
%         Y_com_rec = filter(h_predict,1,x_com_est); % Reconstruct Communication Signal
%         Y_com_rec = reshape(Y_com_rec,[],Ns);
%         Y_com_rec = DFT_matrix * Y_com_rec;
%     end
% end
% %%
% % for j = 1:length(com_power)
% %     RDM = plot_periodogram(E_output_gifdic{j},Ftx_radar,1/100,1/60);
% %     per_title = sprintf('GIFDIC: Pc/Pn = %d',com_power(j));
% %     title(per_title)
% % end
