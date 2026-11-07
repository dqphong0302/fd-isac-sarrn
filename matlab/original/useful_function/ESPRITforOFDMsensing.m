%% ESPRIT for OFDM sensing

function [range,velocity] = ESPRITforOFDMsensing(CIM,k_target)
global lambda Delta_f c Ts
[N,M] = size(CIM);
%% range estimation
z = [CIM(1:N-1,:);CIM(2:N,:)]; % size 2(N-1) x 
R_zz = z*z'/M;                 % size 2(N-1) x 2(N-1)
[U,~,~] = svd(R_zz);           % 2(N-1) diagonal values in decreasing order?
Es = U(:,1:k_target);
Esx = Es(1:N-1,:);
Esy = Es(N:end,:);

EE = [Esx,Esy];               % size (N-1) x 2k_target
[F,~,~] = svd(EE'*EE);        % 2k_target  diagonal values in decreasing order?
F = F(:,end-k_target+1:end);
F1 = F(1:k_target,:);
F2 = F(k_target+1:2*k_target,:);
psi = -F1*inv(F2);
[~,D] = eig(psi);


phi = angle(diag(D));
phi(phi>0) = phi(phi>0);% - 2*pi;
tau = phi/(2*pi*Delta_f);
range = tau*c/2;

%% doppler estimation
z = [CIM(:,1:M-1),CIM(:,2:M)];
R_zz = z.'*conj(z)/N;
[U,~,~] = svd(R_zz);
Es = U(:,1:k_target);
Esx = Es(1:M-1,:);
Esy = Es(M:end,:);

EE = [Esx,Esy];
[F,~,~] = svd(EE'*EE);
F = F(:,end-k_target+1:end);
F1 = F(1:k_target,:);
F2 = F(k_target+1:2*k_target,:);
psi = -F1*inv(F2);
[~,D] = eig(psi);

phi = angle(diag(D));
phi(phi<0) = phi(phi<0);
doppler = phi/(2*pi*Ts);
velocity = doppler*lambda/2;

end

