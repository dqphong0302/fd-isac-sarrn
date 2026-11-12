%% plot_s2fig_pilot_matched.m
% S2 Fig: NMSE versus gamma_c, (a) one pilot symbol, (b) two pilot symbols, every estimator
% with the same number of pilots in each panel. Panel (a) also shows the network of the original
% submission; panel (b) the RLS implementation of the original submission (P never re-initialised).
% Output : results/figures/matlab/S2_Fig.{fig,eps,pdf,png,tif}

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end

%% ---------------- SETTINGS ----------------
csvFile   = fullfile(here, '..', '..', 'results', 'figures', 'nmse_ber_table.csv');
outDir    = fullfile(here, '..', '..', 'results', 'figures', 'matlab');
figName   = 'S2_Fig';
fontName  = 'Arial';
fontSize  = 9;
legSize   = 7.5;
lineW     = 1.2;
markSize  = 5;
figSizeCm = [19 11];        % extra height for the legends below the panels

% key, label, colour, marker, style
panelA = {   % Np = 1
    'ls',        'LS',                              [0.50 0.50 0.50], 'o', '--'
    'rls',       'RLS',                             [0.09 0.75 0.81], 'v', '--'
    'lmmse',     'LMMSE',                           [0.58 0.40 0.74], 'd', '-.'
    'sarrn_v36', 'ResMLP, original submission',     [0.89 0.47 0.76], '^', ':'
    'ddls',      'DD-LS',                           [0.17 0.63 0.17], '>', '-'
    'sarrn_da',  'SARRN',                           [0.84 0.15 0.16], 'p', '-'
    };
panelB = {   % Np = 2
    'ls',         'LS',                             [0.12 0.47 0.71], 's', '--'
    'rls',        'RLS',                            [0.09 0.75 0.81], 'v', '--'
    'rls_carryP', 'RLS, original submission code',  [0.74 0.74 0.13], 'x', ':'
    'lmmse',      'LMMSE',                          [0.58 0.40 0.74], 'd', '-.'
    'ddls',       'DD-LS',                          [0.17 0.63 0.17], '>', '-'
    };
% -------------------------------------------

if ~exist(outDir, 'dir'), mkdir(outDir); end
T = readtable(csvFile);                 % works from R2016b to current releases
T.method = string(T.method);

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 figSizeCm]);
panels = {panelA, panelB}; titles = {'(a) 1 pilot symbol', '(b) 2 pilot symbols'};
hasTiled = exist('tiledlayout', 'builtin') || exist('tiledlayout', 'file');
if hasTiled
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
end
for p = 1:2
    if hasTiled
        ax = nexttile(tl);
    else
        ax = subplot(1, 2, p);
    end
    hold(ax, 'on'); box(ax, 'on');
    C = panels{p};
    for k = 1:size(C, 1)
        d = pick(T, p, C{k,1});
        fill(ax, [d.gamma_db; flipud(d.gamma_db)], [d.nmse_lo_db; flipud(d.nmse_hi_db)], ...
            1 - 0.22 * (1 - C{k,3}), 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
    h = gobjects(size(C, 1), 1);
    for k = 1:size(C, 1)
        d = pick(T, p, C{k,1});
        h(k) = plot(ax, d.gamma_db, d.nmse_db, 'LineStyle', C{k,5}, 'Color', C{k,3}, ...
            'Marker', C{k,4}, 'MarkerFaceColor', 'none', 'MarkerSize', markSize, ...
            'LineWidth', lineW, 'DisplayName', C{k,2});
    end
    title(ax, titles{p}, 'FontWeight', 'normal');
    xlabel(ax, '\gamma_c (dB)'); if p == 1, ylabel(ax, 'NMSE (dB)'); end
    xlim(ax, [0 30]); ylim(ax, [-85 -15]);
    grid(ax, 'on'); ax.GridLineStyle = ':';
    set(ax, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 0.8);
    lg = legend(ax, 'Location', 'southoutside');   % below the panel, so no curve is hidden
    try lg.NumColumns = 2; catch, end               % NumColumns: R2018a or newer
    set(lg, 'FontSize', legSize);
end

exportAll(fig, outDir, figName);
fprintf('Saved %s.{fig,eps,pdf,png,tif} in %s\n', figName, outDir);

%% ---------------- local functions ----------------
function d = pick(T, Np, key)
    d = T(T.np == Np & T.method == key, :);
    if isempty(d), error('No rows for np=%d, method=%s', Np, key); end
    d = sortrows(d, 'gamma_db');
end

function exportAll(fig, outDir, name)
    f = fullfile(outDir, name);
    savefig(fig, [f '.fig']);
    try
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
