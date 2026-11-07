function [sortedTargets, CUT] = cfar_ofdm_radar(Per, numTargets, offset, Tr, Tc, Gr, Gc, plotFlag)
% CFAR detector for OFDM radar periodogram with global variables, limited target output, and optional plotting
% Global variables (assumed defined):
%   Nc: Number of subcarriers (range dimension)
%   Ns: Number of symbols (Doppler dimension)
%   c: Speed of light (m/s)
%   Delta_f: Frequency step (Hz)
%   fC: Carrier frequency (Hz)
%   Ts: Sampling period (s)
% Inputs:
%   Per: Periodogram (2D matrix from 2D FFT)
%   Tr, Tc: Number of training cells in range and Doppler
%   Gr, Gc: Number of guard cells in range and Doppler
%   offset: SNR threshold offset in dB
%   G: Fractional limit for range peak search (e.g., 1/100)
%   D: Fractional limit for Doppler peak search (e.g., 1/60)
%   numTargets: Maximum number of targets to output (sorted by range)
%   plotFlag: Boolean to enable/disable heatmap plotting (default: false)
% Outputs:
%   sortedTargets: Table with up to numTargets rows of range and velocity, sorted by range (ascending)
%   CUT: Binary detection map (1 for detected targets, 0 otherwise)

% Set default for plotFlag
if nargin < 8
    plotFlag = false;
end

% Validate numTargets
if ~isnumeric(numTargets) || numTargets < 0 || floor(numTargets) ~= numTargets
    error('numTargets must be a non-negative integer.');
end

% Declare global variables
global Nc Ns c Delta_f fC Ts;

% Validate global variables
if isempty(Nc) || isempty(Ns) || isempty(c) || isempty(Delta_f) || isempty(fC) || isempty(Ts)
    error('Global variables Nc, Ns, c, Delta_f, fC, Ts must be defined.');
end

% % Define periodogram parameters
% Nper = 4 * Nc; % FFT length for rows (range dimension)
% Mper = 4 * Ns; % FFT length for columns (Doppler dimension)
% 
% % Normalize the periodogram
% Per = Per / max(max(Per));
% [row, col] = size(Per);
% 
% % Initialize binary detection map
% CUT = zeros(row, col);
% 
% % Slide the CUT across the periodogram, avoiding edges
% for i = (Tr + Gr + 1):(row - Tr - Gr)
%     for j = (Tc + Gc + 1):(col - Tc - Gc)
%         noise_level = 0;
%         % Sum noise in training cells, excluding guard cells
%         for k = (i - Tr - Gr):(i + Tr + Gr)
%             for h = (j - Tc - Gc):(j + Tc + Gc)
%                 if abs(k - i) > Gr || abs(h - j) > Gc
%                     noise_level = noise_level + Per(k, h);
%                 end
%             end
%         end
%         % Calculate number of training cells
%         length_window = 2 * (Tr + Gr) + 1;
%         width_window = 2 * (Tc + Gc) + 1;
%         guard_area = (2 * Gr + 1) * (2 * Gc + 1);
%         num_training_cells = length_window * width_window - guard_area;
% 
%         % Compute threshold with offset
%         threshold = (noise_level / num_training_cells) * offset;
% 
%         % Detect targets
%         if Per(i, j) > threshold
%             CUT(i, j) = 1;
%         end
%     end
% end
% CC = bwconncomp(CUT);
% 
% % Create empty result
% BW_reduced = false(size(CUT));
% 
% % For each connected component, keep only one pixel
% for k = 1:CC.NumObjects
%     % pixels = CC.PixelIdxList{k};
%     % Option 1: keep the first pixel found
%     % BW_reduced(pixels(1)) = true;
% 
%     % Option 2 (uncomment to use centroid instead):
%     stats = regionprops(CC, 'Centroid');
%     centroid = round(stats(k).Centroid);
%     BW_reduced(centroid(2), centroid(1)) = true;
% end
% BW_reduced = logical(BW_reduced);
% CUT = double(BW_reduced);
% 
% % Define indices for range and Doppler
% nIndexes = (1:row)';
% mIndexes = (1:col) - (col/2 + 1);
% 
% % Calculate range and velocity vectors
% distancesVec = (nIndexes - 1) * c / (2 * Delta_f * Nper);
% velocitiesVec = mIndexes * c / (2 * fC * Ts * Mper);
% 
% % Find detected targets (where CUT == 1)
% [y, x] = find(CUT == 1);
% 
% % Check if any targets were detected
% if isempty(y)
%     sortedTargets = table([], [], 'VariableNames', {'ranges', 'velocities'});
%     return;
% end
% 
% % Estimate indices, time delay, distance, and velocity for all detected points
% m_hat = x - (col/2 + 1);
% n_hat = row - y + 1;
% distances = (n_hat - 1) * c / (2 * Delta_f * Nper);
% velocities = (m_hat - 1) * c / (2 * fC * Ts * Mper);
% 
% % Create table and sort by range
% targets = table(distances, velocities, 'VariableNames', {'ranges', 'velocities'});
% sortedTargets = sortrows(targets, 'ranges');
% 
% % Limit output to numTargets
% if size(sortedTargets, 1) > numTargets
%     sortedTargets = sortedTargets(1:numTargets, :);
% end

% Define periodogram parameters
Nper = 32 * Nc; % FFT length for rows (range dimension)
Mper = 32 * Ns; % FFT length for columns (Doppler dimension)

% Normalize the periodogram
Per = Per / max(max(Per));
[row, col] = size(Per);

% Initialize binary detection map
CUT = zeros(row, col);

% Slide the CUT across the periodogram, avoiding edges
for i = (Tr + Gr + 1):(row - Tr - Gr)
    for j = (Tc + Gc + 1):(col - Tc - Gc)
        noise_level = 0;
        % Sum noise in training cells, excluding guard cells
        for k = (i - Tr - Gr):(i + Tr + Gr)
            for h = (j - Tc - Gc):(j + Tc + Gc)
                if abs(k - i) > Gr || abs(h - j) > Gc
                    noise_level = noise_level + Per(k, h);
                end
            end
        end
        % Calculate number of training cells
        length_window = 2 * (Tr + Gr) + 1;
        width_window = 2 * (Tc + Gc) + 1;
        guard_area = (2 * Gr + 1) * (2 * Gc + 1);
        num_training_cells = length_window * width_window - guard_area;
        
        % Compute threshold with offset
        threshold = (noise_level / num_training_cells) * offset;
        
        % Detect targets
        if Per(i, j) > threshold
            CUT(i, j) = 1;
        end
    end
end
CC = bwconncomp(CUT);

% Create empty result
BW_reduced = false(size(CUT));

% For each connected component, keep only one pixel
for k = 1:CC.NumObjects
    % pixels = CC.PixelIdxList{k};
    % Option 1: keep the first pixel found
    % BW_reduced(pixels(1)) = true;
    
    % Option 2 (uncomment to use centroid instead):
    stats = regionprops(CC, 'Centroid');
    centroid = round(stats(k).Centroid);
    BW_reduced(centroid(2), centroid(1)) = true;
end
BW_reduced = logical(BW_reduced);
CUT = double(BW_reduced);

% Define indices for range and Doppler
nIndexes = (1:row)';
mIndexes = (1:col) - (col/2 + 1);

% Calculate range and velocity vectors
distancesVec = (nIndexes - 1) * c / (2 * Delta_f * Nper);
velocitiesVec = mIndexes * c / (2 * fC * Ts * Mper);

% Find detected targets (where CUT == 1)
[y, x] = find(CUT == 1);

% Check if any targets were detected
if isempty(y)
    sortedTargets = table([], [], 'VariableNames', {'ranges', 'velocities'});
    return;
end

% Estimate indices, time delay, distance, and velocity for all detected points
m_hat = x - (col/2 + 1);
n_hat = row - y + 1;
distances = (n_hat - 1) * c / (2 * Delta_f * Nper);
velocities = (m_hat - 1) * c / (2 * fC * Ts * Mper);

% Get amplitudes from Per at centroid positions
amplitudes = zeros(length(y), 1);
for idx = 1:length(y)
    amplitudes(idx) = Per(y(idx), x(idx));
end

% Create table and sort by amplitude descending
targets = table(distances, velocities, amplitudes, 'VariableNames', {'ranges', 'velocities', 'amplitudes'});
sortedTargets = sortrows(targets, 'amplitudes', 'descend');
sortedTargets = sortedTargets(:, {'ranges', 'velocities'});

% Limit output to numTargets
if size(sortedTargets, 1) > numTargets
    sortedTargets = sortedTargets(1:numTargets, :);
end

% Plot heatmap if plotFlag is true
if plotFlag
    figure;
    h = heatmap(velocitiesVec, flip(distancesVec), CUT, 'Colormap', jet(200));
    s = struct(h);
    s.XAxis.TickLabelRotation = 90;
    distanceLabels = string(round(distancesVec));
    distanceLabels(~(mod(distancesVec, 10 * c / (2 * Delta_f * Nper)) == 0)) = "";
    velocityLabels = string(round(velocitiesVec));
    velocityLabels(~(mod(velocitiesVec, 10 * c / (2 * fC * Ts * Mper)) == 0)) = "";
    h.YDisplayLabels = flip(distanceLabels);
    h.XDisplayLabels = velocityLabels;
    xlabel('\fontname{Georgia}\bfVelocity (m/s)');
    ylabel('\fontname{Georgia}\bfDistance (m)');
    title('CFAR Detection Heatmap');
end
end