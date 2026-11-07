function R_noise = generate_noise(noise_power_db)
% GENERATE_NOISE Generates complex Gaussian noise with specified power in dB
% Input:
%   noise_power_db - Desired noise power in dB (scalar)
% Global variables (must be defined in workspace):
%   Nc             - Number of frequency steps
%   Ns             - Number of samples
% Output:
%   R_noise        - Complex noise matrix (Nc-by-Ns)

% Declare global variables
global Nc Ns;

% Convert noise power from dB to linear scale
noise_power = 10^(noise_power_db / 10);
noise_power_sqrt = sqrt(noise_power);

% Generate real and imaginary components
real_part = randn(Nc, Ns); % Real component
imag_part = randn(Nc, Ns); % Imaginary component

% Create complex noise with unit power
complex_noise = (real_part + 1i * imag_part) / sqrt(2);
current_power = mean(abs(complex_noise(:)).^2);
complex_noise = complex_noise / sqrt(current_power);

% Scale to desired power
R_noise = complex_noise * noise_power_sqrt;
end