function out = iBILInterl_ldpc(in,M)
% Bit interleaving, Section 5.4.2.2
    in = reshape(in,length(in)/M,M);
    in = in.';
    out = in(:).';
end