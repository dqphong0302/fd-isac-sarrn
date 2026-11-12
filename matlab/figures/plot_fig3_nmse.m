%% plot_fig3_nmse.m
% Fig 3 of the revised manuscript (PLOS ONE, PONE-D-26-41608):
% NMSE versus communication-to-noise ratio gamma_c, all estimators with ONE pilot symbol.
%
% Data   : results/figures/nmse_ber_table.csv (produced by scripts/eval_channel.py, results/fig3_final).
%          One simulation run, 50 test frames per gamma_c, every estimator on the same frames,
%          final (second) IFD-DIC iteration. Columns used: np, method, gamma_db,
%          nmse_db (mean over frames in linear scale, in dB), nmse_lo_db / nmse_hi_db (95% bootstrap CI).
% Output : results/figures/matlab/Fig3_NMSE.fig (editable in MATLAB), .eps (vector), .pdf, .png, .tif (600 dpi)
%
% Method keys in the CSV -> names used in the paper
%   ls       -> LS
%   rls      -> RLS (inverse correlation matrix re-initialised in every frame)
%   lmmse    -> LMMSE (true power-delay profile)
%   ddls     -> DD-LS (decision-directed LS over the whole detected frame, no network)
%   sarrn_da -> SARRN (proposed: DD-LS anchor + ResMLP residual). This is the only "SARRN" in the paper.
% Not plotted here: sarrn_v36 (network of the original submission), ridge (identical to LS),
%   rls_carryP (RLS of the submitted code), sarrn_etilde. They remain in Table 4 / S2 Fig.
%
% Change fonts, sizes, colours or labels in the SETTINGS block only.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end

%% ---------------- SETTINGS ----------------
csvFile   = fullfile(here, '..', '..', 'results', 'figures', 'nmse_ber_table.csv');
outDir    = fullfile(here, '..', '..', 'results', 'figures', 'matlab');
figName   = 'Fig3_NMSE';
Np        = 1;              % number of pilot symbols shown (reviewer: same pilots for every method)
showCI    = true;           % shaded 95% bootstrap confidence bands
yMode     = 'dB';           % 'dB'     : NMSE in dB on a linear axis (numbers quoted in the text)
                            % 'linear' : NMSE on a logarithmic axis, as in Fig 3 of the 9 Nov manuscript
fontName  = 'Arial';
fontSize  = 10;             % axes and labels
legSize   = 9;              % legend
lineW     = 1.4;
markSize  = 6;
figSizeCm = [14 10.5];      % width x height in cm (PLOS: max width 19.05 cm)

% key, legend label, colour (RGB 0-1), marker, line style
curves = {
    'ls',       'LS, 1 pilot',     [0.50 0.50 0.50], 'o', '--'
    'rls',      'RLS, 1 pilot',    [0.09 0.75 0.81], 'v', '--'
    'lmmse',    'LMMSE, 1 pilot',  [0.58 0.40 0.74], 'd', '-.'
    'ddls',     'DD-LS, 1 pilot',  [0.17 0.63 0.17], '>', '-'
    'sarrn_da', 'SARRN, 1 pilot',  [0.84 0.15 0.16], 'p', '-'
    };
% -------------------------------------------

if ~exist(outDir, 'dir'), mkdir(outDir); end
T = readtable(csvFile);                 % works from R2016b to current releases
T.method = string(T.method);
useLinear = strcmpi(yMode, 'linear');
if useLinear   % dB -> linear NMSE
    T.nmse_db    = 10.^(T.nmse_db    / 10);
    T.nmse_lo_db = 10.^(T.nmse_lo_db / 10);
    T.nmse_hi_db = 10.^(T.nmse_hi_db / 10);
end

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 figSizeCm]);
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on');

% 1) confidence bands first, so that every line is drawn on top of them.
%    Opaque "light" colours are used instead of transparency, so the EPS stays a pure vector file.
if showCI
    for k = 1:size(curves, 1)
        d = pick(T, Np, curves{k,1});
        light = 1 - 0.22 * (1 - curves{k,3});
        fill(ax, [d.gamma_db; flipud(d.gamma_db)], [d.nmse_lo_db; flipud(d.nmse_hi_db)], light, ...
            'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

% 2) mean curves
h = gobjects(size(curves, 1), 1);
for k = 1:size(curves, 1)
    d = pick(T, Np, curves{k,1});
    h(k) = plot(ax, d.gamma_db, d.nmse_db, ...
        'LineStyle', curves{k,5}, 'Color', curves{k,3}, 'Marker', curves{k,4}, ...
        'MarkerFaceColor', curves{k,3}, 'MarkerSize', markSize, 'LineWidth', lineW, ...
        'DisplayName', curves{k,2});
end

xlabel(ax, 'Communication-to-noise ratio \gamma_c (dB)');
xlim(ax, [0 30]); xticks(ax, 0:5:30);
if useLinear
    ylabel(ax, 'NMSE');
    set(ax, 'YScale', 'log'); ylim(ax, [1e-9 1e-1]);
else
    ylabel(ax, 'NMSE (dB)');
    ylim(ax, [-85 -15]); yticks(ax, -80:10:-20);
end
grid(ax, 'on'); ax.GridLineStyle = ':'; ax.GridAlpha = 0.5;
set(ax, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 0.8, 'TickDir', 'in');
lg = legend(ax, 'Location', 'southwest');
set(lg, 'FontSize', legSize, 'FontName', fontName);
lg.Box = 'on';

exportAll(fig, outDir, figName);
fprintf('Saved %s.{fig,eps,pdf,png,tif} in %s\n', figName, outDir);

% Print the plotted values (to check against Table 4)
for k = 1:size(curves, 1)
    d = pick(T, Np, curves{k,1});
    fprintf('%-16s', curves{k,2}); fprintf(' %8.3g', d.nmse_db); fprintf('\n');
end

%% ---------------- local functions ----------------
function d = pick(T, Np, key)
    d = T(T.np == Np & T.method == key, :);
    if isempty(d), error('No rows for np=%d, method=%s', Np, key); end
    d = sortrows(d, 'gamma_db');
end

function exportAll(fig, outDir, name)
    f = fullfile(outDir, name);
    savefig(fig, [f '.fig']);                                   % editable MATLAB figure
    try   % R2020a or newer
        exportgraphics(fig, [f '.eps'], 'ContentType', 'vector');
        exportgraphics(fig, [f '.pdf'], 'ContentType', 'vector');
        exportgraphics(fig, [f '.png'], 'Resolution', 300);
        exportgraphics(fig, [f '.tif'], 'Resolution', 600);   % PLOS: 300-600 dpi TIFF
    catch
        set(fig, 'PaperPositionMode', 'auto', 'Renderer', 'painters');
        print(fig, [f '.eps'], '-depsc2');
        print(fig, [f '.pdf'], '-dpdf', '-bestfit');
        print(fig, [f '.png'], '-dpng', '-r300');
        print(fig, [f '.tif'], '-dtiff', '-r600');
    end
end
