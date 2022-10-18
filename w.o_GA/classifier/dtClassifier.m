function res = dtClassifier(X, Y, sim_seq)
if size(sim_seq,2) > size(label,2)
    sim_num = size(label,2);
else
    sim_num = size(sim_seq,2);
end
res = zeros(sim_num, 1);

parfor i = 1:sim_num
temp_mean = 0;

% 	idx = sim_seq(:,i);
% 	tr_data = X(idx,:);
% 	tr_ans = Y(idx,:);
% 	ts_data = X(~idx,:);
% 	ts_ans = Y(~idx,:);

    tr_data = X(sim_seq(:,i),:);
    tr_ans = Y(sim_seq(:,i),:);
    ts_data = X(~sim_seq(:,i),:);
    ts_ans = Y(~sim_seq(:,i),:);

% 	fitctree %???
% 	mdl = fitctree(tr_data ,tr_ans);
    model = fitctree(tr_data ,tr_ans);
%     pre = mdl.predict(ts_data);
    pre = model.predict(ts_data);
    
%     res(i,1) = sum(pre == ts_ans) / size(ts_ans, 1);
    temp = sum(pre == ts_ans) / size(ts_ans, 1);
    for k = 1:sim_num
        temp_mean = temp_mean + temp(k);
    end
    res(i,1) = temp_mean / sim_num;
end
end