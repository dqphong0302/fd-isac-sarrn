%% plot_fig3_nmse.m
% Fig 3 (PLOS ONE PONE-D-26-41608, Reviewer Major comment 4): NMSE versus gamma_c,
% every estimator with the SAME single pilot symbol, on the same test frames.
%
% Data   : data/nmse_table.csv  (np, method, gamma_db, frames, nmse_db, nmse_lo_db, nmse_hi_db, ...)
%          50 test frames per gamma_c, final (second) IFD-DIC iteration, sigma_nr = 0 dB.
%          nmse_db = mean NMSE over frames (linear mean, then dB); nmse_lo/hi = 95% bootstrap interval.
% Output : output/Fig3_NMSE.fig (editable), .eps (vector), .pdf, .png, .tif (600 dpi)
%
% Method keys in the CSV
%   ls         LS from the pilot symbol
%   rls        RLS (inverse correlation matrix re-initialised in every frame)
%   lmmse      LMMSE with the true power-delay profile
%   ridge      ridge estimator of Eq (9) WITHOUT neural refinement (reviewer request)
%   channelnet ChannelNet [19] (SRCNN + DnCNN), re-trained on the same training data
%   sarrn_v36  SARRN (ridge anchor + ResMLP), the trained model of the manuscript
%
% Change fonts, sizes, colours or labels in the SETTINGS block only. Runs on MATLAB R2018a or newer.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end

%% ---------------- SETTINGS ----------------
csvFile   = fullfile(here, 'data', 'nmse_table.csv');
outDir    = fullfile(here, 'output');
figName   = 'Fig3_NMSE';
Np        = 1;              % same number of pilot symbols for every estimator
showCI    = true;           % shaded 95% bootstrap confidence bands
yMode     = 'linear';       % 'linear': NMSE on a logarithmic axis (as Fig 3 of the manuscript)
                            % 'dB'    : NMSE in dB on a linear axis
fontName  = 'Arial';
fontSize  = 10;
legSize   = 9;
lineW     = 1.3;
markSize  = 6;
figSizeCm = [14 10.5];      % width x height in cm (PLOS: max width 19.05 cm)

% key, legend label, colour (RGB 0-1), marker, line style
curves = {
    'ls',         'LS, 1 pilot',             [0.50 0.50 0.50], 'o', '--'
    'rls',        'RLS, 1 pilot',            [0.09 0.75 0.81], 'v', '--'
    'lmmse',      'LMMSE, 1 pilot',          [0.58 0.40 0.74], 'd', '-.'
    'ridge',      'Ridge (no NN), 1 pilot',  [0.55 0.34 0.29], 'x', ':'
    'channelnet', 'ChannelNet, 1 pilot',     [0.12 0.47 0.71], 's', '-.'
    'sarrn_v36',  'SARRN, 1 pilot',          [0.84 0.15 0.16], 'p', '-'
    };
% -------------------------------------------

if ~exist(outDir, 'dir'), mkdir(outDir); end
T = readtable(csvFile);
T.method = string(T.method);
useLinear = strcmpi(yMode, 'linear');
if useLinear
    T.nmse_db    = 10.^(T.nmse_db    / 10);
    T.nmse_lo_db = 10.^(T.nmse_lo_db / 10);
    T.nmse_hi_db = 10.^(T.nmse_hi_db / 10);
end

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 figSizeCm]);
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on');
if useLinear, set(ax, 'YScale', 'log'); end

% confidence bands first (opaque light colours keep the EPS a pure vector file)
if showCI
    for k = 1:size(curves, 1)
        d = pick(T, Np, curves{k,1});
        light = 1 - 0.22 * (1 - curves{k,3});
        fill(ax, [d.gamma_db; flipud(d.gamma_db)], [d.nmse_lo_db; flipud(d.nmse_hi_db)], light, ...
            'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

for k = 1:size(curves, 1)
    d = pick(T, Np, curves{k,1});
    plot(ax, d.gamma_db, d.nmse_db, 'LineStyle', curves{k,5}, 'Color', curves{k,3}, ...
        'Marker', curves{k,4}, 'MarkerSize', markSize, 'LineWidth', lineW, 'DisplayName', curves{k,2});
end

xlabel(ax, 'Communication-to-noise ratio \gamma_c (dB)');
xlim(ax, [0 30]); xticks(ax, 0:5:30);
if useLinear
    ylabel(ax, 'NMSE');
else
    ylabel(ax, 'NMSE (dB)');
end
grid(ax, 'on'); ax.GridLineStyle = ':';
set(ax, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 0.8);
lg = legend(ax, 'Location', 'southwest');
set(lg, 'FontSize', legSize, 'FontName', fontName);

exportAll(fig, outDir, figName);
fprintf('Saved %s.{fig,eps,pdf,png,tif} in %s\n', figName, outDir);

% plotted values, to check against the table in the manuscript
for k = 1:size(curves, 1)
    d = pick(T, Np, curves{k,1});
    if useLinear, v = 10*log10(d.nmse_db); else, v = d.nmse_db; end
    fprintf('%-24s', curves{k,2}); fprintf(' %6.1f', v); fprintf('  (dB)\n');
end

%% ---------------- local functions ----------------
function d = pick(T, Np, key)
    d = T(T.np == Np & T.method == key, :);
    if isempty(d), error('No rows for np=%d, method=%s', Np, key); end
    d = sortrows(d, 'gamma_db');
end

function exportAll(fig, outDir, name)
    f = fullfile(outDir, name);
    savefig(fig, [f '.fig']);
    try   % R2020a or newer
        exportgraphics(fig, [f '.eps'], 'ContentType', 'vector');
        exportgraphics(fig, [f '.pdf'], 'ContentType', 'vector');
        exportgraphics(fig, [f '.png'], 'Resolution', 300);
        exportgraphics(fig, [f '.tif'], 'Resolution', 600);
    catch
        set(fig, 'PaperPositionMode', 'auto', 'Renderer', 'painters');
        print(fig, [f '.eps'], '-depsc2');
        print(fig, [f '.pdf'], '-dpdf', '-bestfit');
        print(fig, [f '.png'], '-dpng', '-r300');
        print(fig, [f '.tif'], '-dtiff', '-r600');
    end
end
