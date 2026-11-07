function [E,T1,Td] = fdic(Rmat, Ftx_radar, H_comp_mat)
% FDIC (Frequency Differential Interference Cancellation) for OFDM radar
% Global variables (assumed defined):
%   Ns: Number of symbols (columns in Rmat and Ftx_radar)
% Inputs:
%   Rmat: Received signal matrix (Nc x Ns)
%   Ftx_radar: Transmitted radar signal matrix (Nc x Ns)
% Outputs:
%   E: Output matrix after FD-CIC processing (Nc x Ns)
%   Tmat: Differential sequence matrix (Nc x Ns) (optional)
%   U: Intermediate matrix after accumulating differential sequence (Nc x Ns) (optional)

% Declare global variable
global Ns;

% Validate inputs
if isempty(Ns) || ~isnumeric(Ns) || Ns < 2 || floor(Ns) ~= Ns
    error('Global variable Ns must be defined as an integer >= 2.');
end
if ~ismatrix(Rmat) || ~ismatrix(Ftx_radar)
    error('Rmat and Ftx_radar must be 2D matrices.');
end
if ~isequal(size(Rmat), size(Ftx_radar))
    error('Rmat and Ftx_radar must have the same size.');
end
if size(Rmat, 2) ~= Ns
    error('Number of columns in Rmat and Ftx_radar must equal Ns.');
end

if (nargin < 3)
    H_comp_mat = zeros(size(Rmat));
end

% Initialize differential sequence matrix
Tmat = zeros(size(Rmat));

% Compute differential sequence Tmat
% Tn,m = Rn,m - (Ftx_radar_n,m / Ftx_radar_n,m+1) * Rn,m+1
for col = 1:(Ns-1)
    Tmat(:, col) = Rmat(:, col) - (Ftx_radar(:, col) ./ Ftx_radar(:, col+1)) .* Rmat(:, col+1) + H_comp_mat(:,col).*Ftx_radar(:,col);
    T1(:,col) = Rmat(:, col) - (Ftx_radar(:, col) ./ Ftx_radar(:, col+1)) .* Rmat(:, col+1);
    Td(:,col) = H_comp_mat(:,col).*Ftx_radar(:,col);
end

% Initialize intermediate matrix U
U = zeros(size(Rmat));
U(:, 1) = Tmat(:, 1); % Un0 = Tn,0
prevU = U(:, 1);

% Compute U matrix
% Un,m = Un,m-1 + (Ftx_radar_n,1 / Ftx_radar_n,m) * Tn,m
for col = 2:Ns
    U(:, col) = prevU + (Ftx_radar(:, 1) ./ Ftx_radar(:, col)) .* Tmat(:, col);
    prevU = U(:, col);
end

% Initialize output matrix E
E = zeros(size(Rmat));
E(:, 1) = (1 / (Ns - 1)) * sum(U(:, 1:(Ns-1)), 2); % En0 = (1/(Ns-1)) * sum(U(:,1:Ns-1), 2)

% Compute E matrix
% En,m = (Ftx_radar_n,m / Ftx_radar_n,1) * (En0 - Un,m-1)
for col = 2:Ns
    E(:, col) = (Ftx_radar(:, col) ./ Ftx_radar(:, 1)) .* (E(:, 1) - U(:, col-1));
end
end