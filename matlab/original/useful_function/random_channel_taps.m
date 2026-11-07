function taps = random_channel_taps(tap_dB_mean, phase_first)
% RANDOM_CHANNEL_TAPS Generate random channel taps
% 
% taps = RANDOM_CHANNEL_TAPS(tap_dB_mean, phase_first)
% 
% Inputs:
%   tap_dB_mean : 1×N vector, average tap magnitudes in dB (first should be 0)
%   phase_first : phase (in radians) of first tap, usually 0
%
% Output:
%   taps : 1×N complex vector of random taps
%
% Notes:
% - First tap magnitude is fixed from tap_dB_mean(1) with given phase
% - Other taps have random magnitudes around given mean (±2 dB) and random phases
% - Magnitudes of taps 2..N are strictly decreasing

    N = numel(tap_dB_mean);
    taps = zeros(1, N);

    % First tap
    taps(1) = db2mag(tap_dB_mean(1)) * exp(1j * phase_first);

    % Other taps: generate random magnitude around mean and random phase
    mags = zeros(1, N-1);
    phases = zeros(1, N-1);
    for k = 2:N
        rand_dB = tap_dB_mean(k) + (randn * 2); % ±2 dB variation
        mags(k-1) = db2mag(rand_dB);
        phases(k-1) = 2*pi*rand;
    end

    % Sort magnitudes in decreasing order
    [sorted_mags, sort_idx] = sort(mags, 'descend');

    % Assign sorted magnitudes with random phases
    taps(2:end) = sorted_mags .* exp(1j * phases(sort_idx));
end
