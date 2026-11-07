function out = iBILInterl(in)
% Triangular interleaver
%
%   OUT = iBILInterl(IN) performs triangular interleaving on the input, IN,
%   writing in the input E elements row-wise and returns the output, OUT,
%   by reading them out column-wise.
%   in here is vector e
%   out here is vector s

%   Reference: TS 38.212, Section 5.4.1.3.

    % Get T off E
    E = length(in);
    T = ceil((-1+sqrt(1+8*E))/2);

    % Write input to buffer row-wise
    v = -1*ones(T,T,class(in));   % <NULL> bits
    k = 0;
    for i = 0:T-1
        for j = 0:T-1-i
            if k < E
                v(i+1,j+1) = in(k+1);
            end
            k = k+1;
        end
    end

    % Read output from buffer column-wise
    out = zeros(size(in),class(in));
    k = 0;
    for j = 0:T-1
        for i = 0:T-1-j
            if v(i+1,j+1) ~= -1 % different with NULL bits
                out(k+1) = v(i+1,j+1);
                k = k+1;
            end
        end
    end

end