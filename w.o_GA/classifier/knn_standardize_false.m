function [avg, std_dev, perf] = knn_standardize_false(data,label, sim_seq)
sim_num=10;
for k=1:sim_num
    tr_data = data( sim_seq(:,k), : );
    tr_ans = label( sim_seq(:,k), : );
    ts_data = data( ~sim_seq(:,k), :);
    ts_ans = label(~sim_seq(:,k), :);
    
    result(k) = knnClassifier_standardize_false(tr_data, tr_ans, ts_data, ts_ans);

end

avg = mean(result);
std_dev = std(result);
perf = result.';
end