clear;

dataName = 'dis_giantsteps_tempo';

data_path = 'C:\Users\CAU_MI\Desktop\개인연구\GAFS2\dataset\dis_900\';
out_path = 'C:\Users\CAU_MI\Desktop\개인연구\GAFS2\nb\scores\';
ext = '.mat';

%load(sprintf('%s%s%s',data_path,dataName,ext), "data_dis", "answer", "sim_seq")
load(sprintf('%s%s%s',data_path,dataName,ext))

exp_iter = 10;

%pctRunOnAll warning( 'off');
   warning( 'off' )
   
scores = cell(10,1);

%parfor k=1:exp_iter
for k=1:exp_iter
    k
    train_data = data( sim_seq(:,k), : );
    train_answer = label( sim_seq(:,k), : );
    score = cell(1,3);
    [score{1,1}, score{1,2}, score{1,3}] = cal_scores(train_data, train_answer);
    scores{k,1} = score;
end

save(sprintf('%s%s%s',out_path,dataName,ext), 'scores')



function [f_ent, fl_ent, ff_ent] = cal_scores(data, answer)

col = size(data,2);
lcol = size(answer,2);

f_ent = zeros( col, 1 );
fl_ent = zeros( col, lcol );

ff_ent = zeros( col, col );
for k=1:col
    f_ent(k,1) = p_entropy( data(:,k) );
    for m=1:lcol
        fl_ent(k,m) = p_entropy( [data(:,k),answer(:,m)] );
    end
end
for k = 1:col
    for m = k:col
        ff_ent(k,m) = p_entropy([data(:,k), data(:,m)]);  
    end
end
ff_ent = ff_ent + ff_ent';
for k = 1:col
    ff_ent(k,k) = ff_ent(k, k) / 2;
end
end


function [res] = p_entropy( vector )

[uidx,~,single] = unique( vector, 'rows' );
count = zeros(size(uidx,1),1);
for k=1:size(vector,1)
    count( single(k), 1 ) = count( single(k), 1 ) + 1;
end
res = -( (count/size(vector,1))'*log2( (count/size(vector,1)) ) );
end