function out = iBILDeinterl(in)
% Triangular deinterleaver
%
%   OUT = iBILDeinterl(IN) performs triangular deinterleaving on the input,
%   IN, and returns the output, OUT.
%
%   Reference: TS 38.212, Section 5.4.1.3.

    % Get T off E
    E = length(in);
    T = ceil((-1+sqrt(1+8*E))/2);;

    % Create the table with nulls (filled in row-wise)
    vTab = zeros(T,T,class(in));
    k = 0;
    for i = 0:T-1
        for j = 0:T-1-i
            if k < E
                vTab(i+1,j+1) = k+1;
            end
            k = k+1;
        end
    end

    % Write input to buffer column-wise, respecting vTab
    v = Inf*ones(T,T,class(in));
    k = 0;
    for j = 0:T-1
        for i = 0:T-1-j
            if k < E && vTab(i+1,j+1) ~= 0
                v(i+1,j+1) = in(k+1);
                k = k+1;
            end
        end
    end

    % Read output from buffer row-wise
    out = zeros(size(in),class(in));
    k = 0;
    for i = 0:T-1
        for j = 0:T-1-i
            if ~isinf(v(i+1,j+1))
                out(k+1) = v(i+1,j+1);
                k = k+1;
            end
        end
    end

end

