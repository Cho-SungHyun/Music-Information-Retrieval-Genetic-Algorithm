function acc = knnClassifier_standardize_false(tr_data, tr_ans, ts_data, ts_ans)

[~,~,tr_ans] = unique( tr_ans, 'rows' );
[~,~,ts_ans] = unique( ts_ans, 'rows' );

model = fitcknn( tr_data, tr_ans, 'NumNeighbors', 3);
pre = model.predict(ts_data);
acc = sum(pre == ts_ans) / size(ts_ans, 1);
end