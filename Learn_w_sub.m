function [output_w, obj_value] = Learn_w_sub(train_data, T, M, L, w, paras)
    % solve w using Accelerated Proximal Gradient
    lambda = paras.lambda2;

    % Neglect features with M-value very small
    output_w = zeros(size(train_data, 2), 1);
    M_value = sum(M.^2, 2);
    neglect_index = M_value <= eps;
    nonzero_index = M_value > eps;
    train_data(:, neglect_index) = [];
    M(neglect_index,:) = []; M(:,neglect_index) = [];
    L(neglect_index,:) = []; w(neglect_index) = [];

    %% compute the optimal w using FISTA
    eta = 1.1;
    t = 1;
    max_iter = 10; 
    nu = 1;
    obj_value = zeros(max_iter+1, 1);
    obj_value(1) = Inf;
    w_x = w;   w_y = w_x;
    for iter = 1:max_iter
        % line search
        [Q_ly_temp, grad_y] = compute_grad2(train_data, M, L, T, w_y);
        while 1
            P_Ly = prox_l1(w_y - (1/nu)*grad_y, lambda/nu);
            F_y = compute_obj2(train_data, L, T, P_Ly);
            Q_ly = Q_ly_temp + grad_y'*(P_Ly - w_y) + 0.5*nu*norm(P_Ly - w_y)^2;
            if F_y <= Q_ly
                break;
            end
            nu = eta * nu;
        end
        w_z = P_Ly;
        t_old = t;
        t = (1 + sqrt(1+4*t^2))/2;

        F_z = F_y + lambda*sum(abs(w_z));
        if F_z < obj_value(iter)
            w_x_old = w_x;
            w_x = w_z;
            obj_value(iter + 1) = F_z;
        else
            w_x_old = w_x;
            obj_value(iter + 1) = obj_value(iter);
        end
        disp(obj_value(iter))

        if iter > 2 && norm(diff(obj_value(iter - 2 : iter + 1))) < 0.001
            break;
        end
        w_y = w_x + (t_old - 1)/t * (w_x - w_x_old) + (t_old / t) * (w_z - w_x);
    end

    function [obj, grad] = compute_grad2(train_data, M, L, T, current_w)
        % evaluate funtion value and gradient for smooth hinge loss
        numT = size(T, 1);
        % project data using current w
        L_hat = diag(current_w) * L;
        train_data_proj = train_data * L_hat;
        [obj_val, obj] = compute_objW_Temp(train_data_proj', T');

        tt = find(abs(obj_val) > eps)';
        W = sparse(T(tt,1), T(tt,3), obj_val(tt), size(train_data,1),  size(train_data,1));
        W = W + sparse(T(tt,1), T(tt,2), -obj_val(tt), size(train_data,1),  size(train_data,1));
        C = SODWkqwAx(train_data', W);
        grad = (M.*C)*current_w / numT;        
    end
    function obj = compute_obj2(train_data, L, T, current_w)
        % evaluate funtion value and gradient for smooth hinge loss
        % project data using current w
        L_hat = diag(current_w) * L;
        train_data_proj = train_data * L_hat;
        obj = compute_objW(train_data_proj', T');
    end

    function x = prox_l1(v, lambda)
        % PROX_L1    The proximal operator of the l1 norm.
        %
        %   prox_l1(v,lambda) is the proximal operator of the l1 norm
        %   with parameter lambda.

        x = max(0, v - lambda) - max(0, -v - lambda);
    end
obj_value = obj_value(1:(iter+1));
output_w(nonzero_index) = w_x;
end