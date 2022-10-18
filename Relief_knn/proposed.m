function [opt_vec, ret_eval] = proposed( IDATA, IANSWER, IPOOL_SIZE,IFFCS, K)
global data
global answer
global pool
global col
global lcol
global row
%global lcons
global acalls
global pool_size
global lacc_mat
%global comp_size
%global comp_fea
%global comp
global N_neighbors

data = IDATA;
answer = IANSWER;
pool_size = IPOOL_SIZE;
ffcs = IFFCS;
acalls = 0;
lcol = size(answer,2);
[row,col] = size( data );
lacc_mat = zeros(pool_size, lcol);
N_neighbors = K;
%comp = ICOMP;
%comp_size = round(lcol * comp);
%comp_fea = round(lcons * comp);

rng(0)

%% Initialize P(t)
% Randomly initialize the pool
% Each chromosome must contain less than LCONS '1' bit
idx = 1:col;
pool = zeros(pool_size,col);
pool(1,:) = 1;
for k=2:pool_size
    tidx = idx;
    %rlen = round(rand()*min((lcons-1),(col-1)))+1;
    %rlen = randi([2,min(lcons,col)]);
    for m=1:randi(col)
        ridx = round(rand()*(length(tidx)-1))+1;
        pool(k,tidx(ridx)) = 1;
        tidx(ridx) = [];
    end
end

% for the case of Multi-label accuracy, pre-allocate 4 cells; [mlacc mlprec mlrec mlf1]
% for the other evaluation measures, values are assigned as NaN
eval = zeros(pool_size, 1);
eval(:,:) = inf;

%% Evaluate P(t)
for k=1:pool_size
    [eval(k,:), lacc_mat(k,:)] = evaluate(pool(k,:)); % Obtaining each fitness
end


%% Sort according to fitness
[eval,sidx] = sortrows(eval,1, 'descend');
pool = pool(sidx,:);
%eval = eval(sidx,:);
lacc_mat = lacc_mat(sidx,:);

opt_vec = pool(1, :);
ret_eval = eval(1, 1);

%% Start Generation
while acalls < ffcs
    while size(pool,1) <= pool_size
        child = [];

        num_rep = 10;

        ts = round(pool_size/num_rep);
        idx = tournamentSelection(ts, eval);

        for k = 1:num_rep
            [sel_idx, lacc_sidx, comp_size] = complement_selection(idx(1,k));
            child(end+1,:) = crossover(idx(k), sel_idx, lacc_sidx, comp_size);
        end
        idx = round(rand()*(size(pool,1)-1))+1;
        child(end+1, :) = mutation(idx);

        for k=1:size(child,1)
            [eval(end+1,:), lacc_mat(end+1,:)] = evaluate(child(k,:)); 
        end
        pool = [pool;child];
    
    
        [~,sidx] = unique( pool, 'rows', 'first' );
        pool = pool(sidx,:);
        eval = eval(sidx,:);
        lacc_mat = lacc_mat(sidx,:);
    end
    
    [eval,sidx] = sortrows(eval,1, 'descend');
    pool = pool(sidx,:);
    lacc_mat = lacc_mat(sidx,:);
    
    lacc_mat = lacc_mat(1:pool_size, :);
    eval = eval(1:pool_size,:);
    pool = pool(1:pool_size,:);
    opt_vec = [opt_vec;pool(1,:)];
    ret_eval = [ret_eval; eval(1,1)];
    
    %fea_cur = sum(pool, 1);
    %zero_sum = find(fea_cur ==0)
    %a = 1;
  
end
end

function selected_idx = tournamentSelection(tournament_size, fitness)

n = length(fitness);
selected_idx = [];
for i = 1:tournament_size  
    shuffleOrder = randperm(n);
    competitors = reshape(shuffleOrder, tournament_size, n/tournament_size)';
    [~, win_idx] = min(fitness(competitors),[],2);
    idMap = (0:tournament_size-1)*n/tournament_size;
    idMap1 = idMap(win_idx) + (1:size(competitors,1));
    selected_idx = [selected_idx; competitors(idMap1)];
end
end

function [sel_idx, lacc_sidx, comp_size] = complement_selection(idx)
global lacc_mat
%global comp_size
global lcol

[slacc_vec, lacc_sidx] = sort(lacc_mat(idx,:), 2);
lacc_dif = zeros(1,lcol-1);
for k=1:lcol-1
    lacc_dif(k) = slacc_vec(k+1) - slacc_vec(k);
end
[~, comp_size] = max(lacc_dif);
comp_size = min([comp_size, round(lcol/2)]);

comp_idx = lacc_sidx(1:comp_size); %잘 못맞추는 label들의 index
lacc_sum = sum(lacc_mat(:,comp_idx), 2); % 모든 크로모좀들에 대한 error label들의 정확도 합 (popsize * 1)

%indices = [1:size(lacc_mat, 1)];
%indices(idx) = [];
%lacc_sum(idx) = [];
%sel_idx = randsample(indices, 1, true, lacc_sum');
[~, sort_idx] = sort(lacc_sum, 1, 'descend');
self_idx = ismember(sort_idx, idx);
sort_idx(self_idx) = []; % 본인 빼고 가장 높은거 뽑음

sel_idx = sort_idx(1);
end

function child = crossover(idx1, idx2, lacc_sidx, comp_size)
global pool
global col
%global lcons
global data
global answer
%global comp_fea
%global lcol
global N_neighbors

%comp = comp_size/lcol;
%comp_fea = round(lcons * comp);
non_comp_idx = lacc_sidx(comp_size+1:end);
comp_idx = lacc_sidx(1:comp_size); 

chr1 = pool(idx1,:);
chr2 = pool(idx2,:);

sel_idx = find((chr2-chr1)==1);
unique_fea_size = size(sel_idx, 2);
if unique_fea_size == 0
    child = mutation(idx1);
    return;
end

del_size = randi(unique_fea_size);
add_size = randi(unique_fea_size);
%change_size = min(round(unique_fea_size*(1-comp)), comp_fea);

chr1_fea_idx = find(chr1);

data_fs = data(:, chr1==1);

cur_f_num = size(data_fs,2);
%fea_idx = gics( data_fs, answer(:, non_comp_idx), cur_f_num-del_size, false, 0, if_ent, iff_ent, ifl_ent);
fea_idx = relieff_fs( data_fs, answer(:, non_comp_idx), cur_f_num-del_size, N_neighbors);

chr1_sel_idx = chr1_fea_idx(fea_idx);

chr2_fea_idx = find(chr2);
data_fs = data(:, sel_idx);
fea_idx = relieff_fs( data_fs, answer(:, comp_idx), add_size, N_neighbors);
add_sel_idx = chr2_fea_idx(fea_idx);

final_idx = union(chr1_sel_idx,add_sel_idx);

%chr_union = zeros(2, col);
%chr_union(1, chr1_sel_idx) = 1; 
%chr_union(2, chr1_sel_idx) = 1;
%chr_union(1, sel_idx) = 1;
%chr_union(2, sel_idx) = 2;
%union_idx = find(chr_union(1, :)==1); 
%chr_union_fs = chr_union(:, union_idx);
%chr1_unique = find(chr_union_fs(2, :) == 1);
%data_fs = data(:, chr_union(1, :)==1);
%fea_idx = gics(data_fs, answer(:, comp_idx), cur_f_num+add_size, true, chr1_unique, if_ent, iff_ent, ifl_ent);

%final_idx = union_idx(fea_idx);


child = zeros(1, col);
child(1, final_idx) = 1;

end

%% Mutation
function child = mutation(idx)
% Complementary Mutation
global pool
%global comp
global lacc_mat
global col
global lcol
%global lcons
global data
global answer
global N_neighbors

[slacc_vec, lacc_sidx] = sort(lacc_mat(idx,:), 2);
lacc_dif = zeros(1,lcol-1);
for k=1:lcol-1
    lacc_dif(k) = slacc_vec(k+1) - slacc_vec(k);
end
[~, comp_size] = max(lacc_dif);
comp_size = min([comp_size, round(lcol/2)]);

%comp = comp_size/lcol;
%comp_fea = round(lcons * comp);
non_comp_idx = lacc_sidx(comp_size+1:end);
comp_idx = lacc_sidx(1:comp_size); 

chr_fea_idx = find(pool(idx,:));
pool_fea_idx = find(sum(pool,1) ~= 0); %새로운 피쳐들 중 유입
cand_fea_idx = setdiff(1:col, pool_fea_idx);
%cand_fea_idx = setdiff(1:col, chr_fea_idx);

cand_fea_size = size(cand_fea_idx,2);
if cand_fea_size == 0
    child = zeros(1,col);
    tidx = 1:col;
    for m=1:randi(col)
        ridx = round(rand()*(length(tidx)-1))+1;
        child(1,tidx(ridx)) = 1;
        tidx(ridx) = [];
    end
    return;
end



%change_size = randi(cand_fea_size);
%change_size = min(round(cand_fea_size*(1-comp)), comp_fea);

del_size = randi(cand_fea_size);
add_size = randi(cand_fea_size);

data_fs = data(:, chr_fea_idx);

cur_f_num = size(data_fs,2);
%fea_idx = gics( data_fs, answer(:, non_comp_idx), cur_f_num-del_size, false, 0, if_ent, iff_ent, ifl_ent); 
fea_idx = relieff_fs( data_fs, answer(:, non_comp_idx), cur_f_num-del_size, N_neighbors);

chr_sel_idx = chr_fea_idx(fea_idx);


data_fs = data(:, cand_fea_idx);
fea_idx = relieff_fs( data_fs, answer(:, comp_idx), add_size, N_neighbors);
add_sel_idx = cand_fea_idx(fea_idx);

final_idx = union(chr_sel_idx,add_sel_idx);

child = zeros(1, col);
child(1, final_idx) = 1;

%pfl_ent = fl_ent(cand_fea_idx, comp_idx);
%w = sum(pfl_ent,2);

%[~,~,comp_ans] = unique( answer(:, comp_idx), 'rows' );
%[~, w] = relieff(data(:, cand_fea_idx), comp_ans, N_neighbors);

%cand_sel_idx = datasample(cand_fea_idx, add_size, 'Replace', false, 'Weights', w');

%child = zeros(1,col);
%child(1, union(chr_sel_idx, cand_sel_idx)) = 1;

end

%% Eval

function [val, lacc] = evaluate( chr )
% Increase the number of actual fitness function calls

global acalls
acalls = acalls + 1;
if all(chr==0)
    val = inf;
    return;
end

global data
global row
global answer
global N_neighbors

[train,test] = crossvalind( 'holdout', ones(row,1), 0.2 );
pre = mcknn_matlab( data(train,chr==1), answer(train,:), data(test,chr==1), N_neighbors);

inter = answer(test,:) == pre;
lacc = sum(inter,1) / size(pre,1);

val = knn_matlab(data(train,chr==1), answer(train,:), data(test,chr==1), answer(test,:), N_neighbors);
end

%% NB for each class
function pre = mcknn_matlab( train_x, train_y, test_x, N_neighbors)

cNum = size(train_y,2);
pre = zeros( size(test_x,1), cNum );

for k=1:cNum
    model = fitcknn( train_x, train_y(:,k), 'NumNeighbors', N_neighbors, 'Distance', 'cosine','Standardize', 1 );
    [pre(:,k),~] = predict( model, test_x  );
    pre(isnan(pre),k) = 0;
end
end