function [Per, distance, velocity] = calc_periodogram(R_nm, Ftx_radar, G, D, scale)
% CALC_PERIODOGRAM Calculates the periodogram for radar target estimation
% Inputs:
%   R_nm           - Received signal matrix (Nc-by-Ns)
%   Ftx_radar      - Transmitted radar signal (complex vector, Nc-by-1)
%   G              - Fraction of OFDM symbol used as guard interval (TG/T)
%   D              - Doppler shift scaling factor
%   scale          - 'non-log' (default) or 'log'
% Globals:
%   Nc, Ns, fC, c, Delta_f, Ts
% Outputs:
%   Per            - Periodogram matrix
%   distance       - Estimated target distance (m)
%   velocity       - Estimated target velocity (m/s)

% Declare global variables
global Nc Ns fC c Delta_f Ts;

% Default scale
if nargin < 5 || isempty(scale)
    scale = 'non-log';
end

% Validate scale
if ~ismember(scale, {'non-log', 'log'})
    error('Scale must be ''non-log'' or ''log''');
end

% FFT parameters
Nper = 32 * Nc; 
Mper = 32 * Ns; 
Nmax = round(G * Nper); 
Mmax = round(D * Mper); 

% Windowing
[numRows, numCols] = size(R_nm);
hamming_2d = hamming(numRows) * hamming(numCols)';
E_windowed = R_nm .* hamming_2d;

% Normalize by transmitted signal
Cper = E_windowed ./ Ftx_radar;

% FFT/IFFT
Cper = Cper';
Cper = ifft(Cper, Mper); 
Cper = Cper';
Cper = fft(Cper, Nper); 
numCols = size(Cper, 2);
Cper = Cper(1:Nmax, :);
Cper = flip(fftshift(Cper, 2), 1); 
Cper = Cper(:, (numCols/2)-Mmax:(numCols/2)+Mmax-1);

% Periodogram calculation
Per = 1/(Nmax*(2*Mmax + 1)) * (abs(Cper).^2);
if strcmp(scale, 'log')
    Per = 10 * log10(Per + eps); 
end

% Find maximum value and its indices
maxPer = max(Per(:));
[y, x] = ind2sub(size(Per), find(Per == maxPer));

% Estimate indices, time delay, distance, and velocity
m_hat = x - (size(Per, 2)/2 + 1);
n_hat = size(Per, 1) - y + 1;
distance = (n_hat - 1) * c / (2 * Delta_f * Nper);
velocity = (m_hat - 1) * c / (2 * fC * Ts * Mper);

end