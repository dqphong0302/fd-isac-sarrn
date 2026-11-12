%% plot_fig6_ber.m
% Fig 6 of the revised manuscript: BER versus gamma_c, one pilot symbol, with 95% Clopper-Pearson intervals.
% Data   : results/figures/nmse_ber_table.csv (bits = frames x (512 - Np) x 1024 per point; errors pooled over frames).
% Output : results/figures/matlab/Fig6_BER.{fig,eps,pdf,png,tif}
% Points with zero bit errors are shown by their one-sided 95% upper bound (down triangle, no line).

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end

%% ---------------- SETTINGS ----------------
csvFile   = fullfile(here, '..', '..', 'results', 'figures', 'nmse_ber_table.csv');
outDir    = fullfile(here, '..', '..', 'results', 'figures', 'matlab');
figName   = 'Fig6_BER';
Np        = 1;
fontName  = 'Arial';
fontSize  = 10;
legSize   = 9;
lineW     = 1.3;
markSize  = 6;
figSizeCm = [14 10.5];

curves = {
    'ls',       'LS, 1 pilot',     [0.50 0.50 0.50], 'o', '--'
    'rls',      'RLS, 1 pilot',    [0.09 0.75 0.81], 'v', '--'
    'lmmse',    'LMMSE, 1 pilot',  [0.58 0.40 0.74], 'd', '-.'
    'ddls',     'DD-LS, 1 pilot',  [0.17 0.63 0.17], '>', '-'
    'sarrn_da', 'SARRN, 1 pilot',  [0.84 0.15 0.16], 'p', '-'
    'genie',    'Perfect CSI',     [0 0 0],          'none', ':'
    };
% -------------------------------------------

if ~exist(outDir, 'dir'), mkdir(outDir); end
T = readtable(csvFile);                 % works from R2016b to current releases
T.method = string(T.method);

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 figSizeCm]);
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on');
set(ax, 'YScale', 'log');

h = gobjects(size(curves, 1), 1);
for k = 1:size(curves, 1)
    d  = pick(T, Np, curves{k,1});
    nz = d.bit_errors > 0;
    h(k) = errorbar(ax, d.gamma_db(nz), d.ber(nz), d.ber(nz) - d.ber_lo(nz), d.ber_hi(nz) - d.ber(nz), ...
        'LineStyle', curves{k,5}, 'Color', curves{k,3}, 'Marker', curves{k,4}, ...
        'MarkerSize', markSize, 'LineWidth', lineW, 'CapSize', 3, 'DisplayName', curves{k,2});
    if any(~nz)   % zero errors: one-sided 95% upper bound
        plot(ax, d.gamma_db(~nz), d.ber_hi(~nz), 'v', 'Color', curves{k,3}, ...
            'MarkerSize', markSize - 2, 'HandleVisibility', 'off');
    end
end

xlabel(ax, 'Communication-to-noise ratio \gamma_c (dB)');
ylabel(ax, 'BER');
xlim(ax, [0 30]); xticks(ax, 0:5:30); ylim(ax, [1e-8 1]);
grid(ax, 'on'); ax.GridLineStyle = ':'; ax.YMinorGrid = 'on';
set(ax, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 0.8);
lg = legend(ax, 'Location', 'southwest');
set(lg, 'FontSize', legSize, 'FontName', fontName);
text(ax, 0.98, 0.97, '\nabla: zero errors, 95% upper bound', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontName', fontName, 'FontSize', legSize - 1);

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
