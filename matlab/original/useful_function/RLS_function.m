function [a_RLS,P] = RLS_function(a_RLS,y_RLS,x,P)
    lambda = 0.999; % Forgetting factor
    alpha = y_RLS - a_RLS'*x;
    g = (lambda^-1*P*x) / (1+x'*lambda^-1*P*x);
    P = (1/lambda)*(P-g*x'*P);
    a_RLS = a_RLS + g*conj(alpha);
end
% function [a_RLS,P] = DF_RLS_function(a_RLS,y_RLS,x,P)
%     lambda = 0.9; % Forgetting factor
%     alpha = y_RLS - (a_RLS.')*x;
%     g = (lambda^-1*P*x) / (1+x'*lambda^-1*P*x);
%     P = (1/lambda)*(P-g*x'*P);
%     a_RLS = a_RLS + g*conj(alpha);
% end