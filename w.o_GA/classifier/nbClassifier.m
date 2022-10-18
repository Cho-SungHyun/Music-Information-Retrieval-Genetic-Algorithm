function acc = nbClassifier(tr_data, tr_ans, ts_data, ts_ans)
warning('off','all')
% acc = zeros(sim_num, 1);

[~,~,tr_ans] = unique( tr_ans, 'rows' );
[~,~,ts_ans] = unique( ts_ans, 'rows' );
    
model = fitcnb(tr_data, tr_ans, 'DistributionName', 'mvmn');
pre = model.predict(ts_data);
acc = sum(pre == ts_ans) / size(ts_ans, 1);
end