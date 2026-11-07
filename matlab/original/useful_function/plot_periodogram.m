function [Per, distance, velocity] = plot_periodogram(R_nm, Ftx_radar, G, D, scale)
% PLOT_PERIODOGRAM Generates and plots a periodogram for radar target estimation
% Inputs:
%   R_nm           - Received signal matrix (Nc-by-Ns)
%   Ftx_radar      - Transmitted radar signal (complex vector, Nc-by-1)
%   G              - Fraction of OFDM symbol used as guard interval (TG/T)
%   D              - Doppler shift scaling factor
%   scale          - Plot scale: 'non-log' (default) or 'log'
% Global variables (must be defined in workspace):
%   Nc             - Number of frequency steps
%   Ns             - Number of samples
%   fC             - Carrier frequency (Hz)
%   c              - Speed of light (m/s)
%   Delta_f        - Frequency step (Hz)
%   Ts             - Sampling period (s)
% Outputs:
%   Per            - Periodogram matrix
%   distance       - Estimated target distance (m)
%   velocity       - Estimated target velocity (m/s)

% Declare global variables
global Nc Ns fC c Delta_f Ts;

% Set default scale to non-log if not provided
if nargin < 5
    scale = 'non-log';
end

% Validate scale input
if ~ismember(scale, {'non-log', 'log'})
    error('Scale must be ''non-log'' or ''log''');
end

% Define periodogram parameters
Nper = 4 * Nc; % FFT length for rows
Mper = 4 * Ns; % FFT length for columns
Nmax = round(G * Nper); % Max row index for peak search
Mmax = round(D * Mper); % Max column index for peak search

% Get size of received signal matrix
[numRows, numCols] = size(R_nm);

% Generate 2D Hamming window
hamming_row = hamming(numRows);
hamming_col = hamming(numCols);
hamming_2d = hamming_row * hamming_col';

% Apply window to received signal
E_windowed = R_nm .* hamming_2d;

% Normalize by transmitted signal
Cper = E_windowed ./ Ftx_radar;

% Compute periodogram
Cper = Cper';
Cper = ifft(Cper, Mper); % IFFT along rows
Cper = Cper';
Cper = fft(Cper, Nper); % FFT along columns
numCols = size(Cper, 2);
Cper = Cper(1:Nmax, :); % Crop rows
Cper = flip(fftshift(Cper, 2), 1); % Shift and flip
Cper = Cper(:, (numCols/2)-Mmax:(numCols/2)+Mmax-1); % Crop columns

% Calculate periodogram
Per = 1/(Nmax*(2*Mmax + 1)) * (abs(Cper).^2);

% Apply logarithmic scale if specified
if strcmp(scale, 'log')
    Per = 10 * log10(Per + eps); % Add eps to avoid log(0)
end

% Find maximum value and its indices
maxPer = max(Per(:));
[y, x] = ind2sub(size(Per), find(Per == maxPer));

% Estimate indices, time delay, distance, and velocity
m_hat = x - (size(Per, 2)/2 + 1);
n_hat = size(Per, 1) - y + 1;
tau_hat = (n_hat - 1) / (Nper * Delta_f);
distance = (n_hat - 1) * c / (2 * Delta_f * Nper);
velocity = (m_hat - 1) * c / (2 * fC * Ts * Mper);

% Generate axis labels
mIndexes = (1:size(Per, 2)) - (size(Per, 2)/2 + 1);
nIndexes = (1:size(Per, 1));
distancesVec = (nIndexes - 1) * c / (2 * Delta_f * Nper);
velocitiesVec = (mIndexes - 1) * c / (2 * fC * Ts * Mper);

% Create label strings for heatmap
distanceLabels = string(round(distancesVec));
distanceLabels(~(mod(distancesVec, 10 * c / (2 * Delta_f * Nper)) == 0)) = "";
velocityLabels = string(round(velocitiesVec));
velocityLabels(~(mod(velocitiesVec, 10 * c / (2 * fC * Ts * Mper)) == 0)) = "";

% Plot heatmap
figure;
h = heatmap(velocitiesVec, flip(distancesVec), Per, 'Colormap', jet(200));
s = struct(h);
s.XAxis.TickLabelRotation = 90;
h.YDisplayLabels = flip(distanceLabels);
h.XDisplayLabels = velocityLabels;
xlabel('\fontname{Georgia}\bfVelocity (m/s)');
ylabel('\fontname{Georgia}\bfDistance (m)');

if strcmp(scale, 'log')
    title('Periodogram (Log Scale)');
else
    title('Periodogram (Non-Log Scale)');
end
end