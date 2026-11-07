function out = iBILDeinterl(in,Qm)
% Perform bit de-interleaving according to TS 38.212 5.4.2.2
    E = length(in);
    in = reshape(in,Qm,E/Qm);
    in = in.';
    out = in(:);
end