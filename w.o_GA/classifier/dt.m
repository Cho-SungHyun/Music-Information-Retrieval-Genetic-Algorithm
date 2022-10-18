function [ans1, ans2] = dt(data,label, sim_seq)
% sim_num = 10;
% sample_num = size(data,1);
% portion = 0.2;

% sim_seq = SeqGen(sim_num, sample_num, portion);

result = dtClassifier(data, label, sim_seq);

ans1 = mean(result);
ans2 = std(result);
end