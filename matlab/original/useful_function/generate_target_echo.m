function Y_target = generate_target_echo(alpha_target_db, v_target, r_target, Ftx_radar)
% GENERATE_TARGET_ECHO Generates summed target echo signal for a radar system
% Inputs:
%   alpha_target_db - Vector of target amplitudes in dB
%   v_target       - Vector of target velocities (m/s)
%   r_target       - Vector of target ranges (m)
%   Ftx_radar      - Transmitted radar signal (complex vector)
% Global variables (must be defined in workspace):
%   Nc             - Number of frequency steps
%   Ns             - Number of samples
%   fC             - Carrier frequency (Hz)
%   c              - Speed of l ight (m/s)
%   Delta_f        - Frequency step (Hz)
%   Ts             - Sampling period (s)
%   Te             - Effective sampling period (s)
% Output:
%   Y_target       - Summed target echo signal (matrix)

% Declare global variables
global Nc Ns fC c Delta_f Ts Te;

% Initialize output as zero matrix with appropriate size
Y_target = zeros(Nc, Ns);

% Convert amplitude from dB to linear scale
alpha_target = sqrt(10.^(alpha_target_db./10));

% Generate DFT matrix
n = 0:Nc-1;
k = n';
DFT_matrix = 1/sqrt(Nc) * exp(-2i * pi * k * n / Nc);

% Loop over each target and sum the echo signals
for i = 1:length(alpha_target_db)
    % Calculate Doppler frequency
    fd_target = 2 * fC * v_target(i) / c;
    
    % Calculate time delay
    tau_target = 2 * r_target(i) / c;
    
    % Time delay vector
    Tl_target = exp(-1j * 2 * pi * Delta_f * tau_target);
    vtl_target = Tl_target .^ (0:Nc-1)';
    
    % Doppler frequency vector
    Fl_target = exp(1j * 2 * pi * fd_target * Ts);
    vfl_target = Fl_target .^ (0:Ns-1)';
    
    % Doppler diagonal matrix
    Dl_target = exp(1j * 2 * pi * fd_target * Te / Nc) .^ (0:Nc-1);
    Dfd_target = diag(Dl_target);
    
    % Compute and accumulate target echo signal
    Y_target = Y_target + alpha_target(i) * Dfd_target * DFT_matrix' * (Ftx_radar .* (vtl_target * vfl_target'));
end
end