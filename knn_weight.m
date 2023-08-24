function acc = knn_weight(data, y, para, cv)
    %y_pred = zeros(cv.NumTestSets, 3);
    cv_acc = zeros(1, 5);
     for i= 1: cv.NumTestSets
        idx = cv.test(i);
        indexTest = find(idx ~= 0);
        indexTrain = find(idx == 0);

        xTrain = data(indexTrain, :);
        xTest = data(indexTest, :);
        yTrain = y(indexTrain);
        yTest = y(indexTest);

        model = FARM(xTrain, yTrain, para);
        newTrain = xTrain * model.L;
        newTest = xTest * model.L;
        if (isempty (model.L) == 1)
            continue;
        end
        Mdl = fitcknn(newTrain, yTrain, 'NumNeighbors', 2);
        %y_pred(i, 3) = predict(Mdl, newTest);
        %y_pred(i, 1) = indexTest;
        %y_pred(i, 2) = yTest;
        result = predict(Mdl, newTest);
        cv_acc(i) = sum(result ==  yTest);
        fprintf('Cross validation done %d', i)
    end
    %acc = sum(y_pred(:, 2)== y_pred(:, 3));
    acc = sum(cv_acc) / length(y);
end

