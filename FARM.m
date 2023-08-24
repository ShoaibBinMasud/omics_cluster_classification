function model = FARM(train_data, train_label, para)
% training feature weighted based metric learning
% parameter for ||M||_F
if ~isfield(para, 'lambda1')
    lambda1 = 1;
    para.lambda1 = lambda1;
else
    lambda1 = para.lambda1;
end
% sparse parameter ||w||_1
if ~isfield(para, 'lambda2')
    lambda2 = 0.01;
    para.lambda2 = lambda2;
else
    lambda2 = para.lambda2;
end
para.update_last_loss_M = 1;
T = generate_knntriplets(train_data, train_label, 3, 10); % for pre post % previous was 3
%mission
%T = generate_knntriplets(train_data, train_label, 2, 1);% for stress shoot
dim = size(train_data, 2);
max_iter = 1; % normally 3
obj_value = zeros(2*max_iter, 1);
w = ones(dim, 1); M = eye(dim); L = M;
if size(T,1) ~= size(train_data)*3*10
    model.w = w;
    model.M = M;
    model.L = L;
    return;
end

%% Alternative Optimization
for iter = 1:max_iter
    % solve w with MFISTA
    [w, fun_val] = Learn_w_sub(train_data, T, M, L, w, para);
    obj_value(2*iter-1) = fun_val(end) + lambda1/2 * norm(M, 'fro')^2;
    
    % fix w and solve M
    proj_train_data = bsxfun(@times, train_data, w');
    [M, L, fun_val] = Learn_M_sub3(proj_train_data, T, w, para);   
    obj_value(2*iter) = fun_val(end) + lambda2 * sum(abs(w));
    
    if iter > 3 && norm(diff(obj_value(iter-3:iter))) < 0.001
        break;
    end
end

% get model output
model.w = w;
model.obj_value = obj_value;
model.M = M;
model.L = L;