function acc = knnClassifier(tr_data, tr_ans, ts_data, ts_ans)
% acc = zeros(sim_num, 1);

[~,~,tr_ans] = unique( tr_ans, 'rows' );
[~,~,ts_ans] = unique( ts_ans, 'rows' );

% model = fitcknn( tr_data, tr_ans, 'NumNeighbors', 3, 'Standardize', 1);
model = fitcknn( tr_data, tr_ans, 'NumNeighbors', 3, 'Distance', 'cosine','Standardize', 1 );

pre = model.predict(ts_data);
pre(isnan(pre),1) = 0;
acc = sum(pre == ts_ans) / size(ts_ans, 1);

% acc(i,1) = temp_mean / sim_num;
end