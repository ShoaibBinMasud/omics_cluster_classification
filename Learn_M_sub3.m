function [output_metric, output_L, fun_val] = Learn_M_sub3(train_data, T, w, para)
% general Distance Metric Learning with smoothed hinge loss
% \min \sum_{t=1}^{T}{\ell(A_t, M)} + 0.5*\lambda*||M||_F s.t. M >= 0

% the output huge metric
output_metric = zeros(length(w), length(w));
% neglect features with zero feature weights
neglect_index = w == 0;
train_data(:, neglect_index) = [];
nonzero_index = find(w ~= 0);

MAX_ITER = 100; % change this value
ABSTOL = 1e-3;
eta = para.lambda1;
L = 1/(size(train_data,1));
M = eye(size(train_data,2));

a = M(:);   aap = zeros(size(a));
t=1;    tp=0;
fun_val = zeros(MAX_ITER, 1);
stop_flag = false;
delta = 1e-6;
debug = false;
%% Nesterov's method (To slove a)
for iter = 1 : MAX_ITER
    % --------------------------- step 1 ---------------------------
    % compute search point s based on aalphap and a (with beta)
    beta = (tp-1)/t;    sa = a + beta * aap;
    
    % --------------------------- step 2 ---------------------------
    % line search for L and compute the new approximate solution x
    % compute the gradient and function value at s
    % get the projection from sa
    [V, D] = eig(symmetric(mat(sa))); Q = V * diag(sqrt(diag(D)));
    train_data_proj = train_data * Q; Q = Q * Q';
    [f_s, obj_val] = compute_objM2(train_data_proj', T');
    W = sparse(T(:,1), T(:,3), obj_val, size(train_data,1), size(train_data,1));
    W = W + sparse(T(:,1), T(:,2), -obj_val, size(train_data,1), size(train_data,1));
    gd_a = vec(SODWkqwAx(train_data', W));
    f_s = f_s + eta/2 * norm(Q,'fro')^2;
    ap = a;
    
    while true
        % let sa walk in a step in the anti-gradient of sa to get va
        % and project va onto the line
        va = (1-eta/L)*sa - gd_a/L;
        [V,D] = eig(symmetric(mat(va)));
        D = diag(D);
        D(D < 1e-10) = 0;
        a = vec(V * diag(D) * V');
        train_data_proj = train_data * (V * diag(sqrt(D)));
        f_new = compute_objM2(train_data_proj', T');
        f_new = f_new + eta/2 * norm(a,2)^2;
        df_a = a - sa;
        r_sum = df_a'*df_a;
        if (sqrt(r_sum) <= delta)
            if debug
                fprintf('\n The distance between the searching point and the approximate is %e, and is smaller than %e \n',...
                    sqrt(r_sum), delta);
            end
            stop_flag = true;
            break;
        end
        
        l_sum = f_new - f_s - (gd_a + eta * sa)'*df_a;
        % the condition is l_sum <= L * r_sum
        if(l_sum <= r_sum*L*0.5)
            break;
        else
            L=2*L;
        end
    end
    
    % --------------------------- step 3 ---------------------------
    % update a and q, and check whether converge
    tp = t;   t = (1+sqrt(4*t*t+1))/2;
    aap = a - ap;
    fun_val(iter) = f_new;
    disp(f_new)
    % ----------- check the stop condition
    if ( (iter >=10) && ...
            abs(fun_val(iter) - fun_val(iter-1)) <= abs(fun_val(iter-1))* ABSTOL)
        if debug
            fprintf('\n Relative obj. gap is less than %e \n',ABSTOL);
        end
        stop_flag=1;
    end
    
    if stop_flag
        break;
    end
    if debug
        fprintf('%d : %f  %d\n', iter, fun_val(iter), L);
    end
end
fun_val = fun_val(1:iter);

% get output metric, put small metric in the large one
index1 = nonzero_index * ones(1,length(nonzero_index));

M = symmetric(mat(va));
[V,D] = eig(M);
D = diag(D);
D(D < 1e-10) = 0;
M = V * diag(D) * V';
trunc_D = sqrt(D);
L_M = V(:, trunc_D ~= 0) * diag(trunc_D(trunc_D ~= 0));
output_L = zeros(size(output_metric,1), size(L_M,2));
output_L(nonzero_index, :) = L_M;
output_metric(sub2ind(size(output_metric), vec(index1'), index1(:))) = M;

    function output = vec(X)
        output = X(:);
    end

    function output = symmetric(X)
        output = 0.5*(X + X');
    end
end