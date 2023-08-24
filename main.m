%read positive mode data

M = readmatrix('conor_pos_sub.csv');
X = M(1:end-1, 2: end)';
y = M(end, 2:end)';
x_tsne = tsne(X, 'Algorithm','exact');
gscatter(x_tsne(:  ,1),x_tsne(:,2 ), y)
Mdl = fitcknn(X, y, 'NumNeighbors', 4);
result = predict(Mdl, X);
sum(result== y)
figure(2)
%%
best_lambda1_1 = 0.0005; best_lambda2_1 = 0.005;
para.lambda1 = best_lambda1_1; para.lambda2 = best_lambda2_1;
model = FARM(X, y, para);
%%
train_data_new = X * model.L;
x_new_tsne = tsne(train_data_new);
gscatter(x_new_tsne(:  ,1),x_new_tsne(:,2 ), new_Y)
Mdl = fitcknn(train_data_new, y, 'NumNeighbors', 4);
result_metric = predict(Mdl, train_data_new);
sum(result_metric== y)