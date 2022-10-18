function [avg, std_dev, perf] = knn(data,label, sim_seq)
sim_num=10;
for k=1:sim_num
    tic;
    k
    tr_data = data( sim_seq(:,k), : );
    tr_ans = label( sim_seq(:,k), : );
    ts_data = data( ~sim_seq(:,k), :);
    ts_ans = label(~sim_seq(:,k), :);
    
    result(k) = knnClassifier(tr_data, tr_ans, ts_data, ts_ans);
    toc;
end

avg = mean(result);
std_dev = std(result);
perf = result.';
end