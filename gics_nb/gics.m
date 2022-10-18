function res = gics( data, answer, numFeat, add, subset , f_ent, ff_ent, fl_ent)

fcol = size( data, 2 );
lcol = size( answer, 2 );

% Determining the first feature
sfl_ent = sum(fl_ent, 2);
trel = sortrows([(1:fcol)', lcol * f_ent - sfl_ent],2 ,'descend');
if add
    res = subset';
else
    res= trel(1,1);
end


for k=(size(res, 1) + 1):numFeat
    trel = zeros(fcol,1);
    for m=1:fcol
        if ismember(m, res)
            trel(m,1) = -inf;
            continue;
        end
        if f_ent(m,1) == 0
            trel(m,1) = -inf;
            continue;
        end
       
        trel(m,1) = lcol * sum(ff_ent(m, res), 2) - size(res, 1) * sfl_ent(m,1);
    end
    scr = sortrows( [(1:fcol)',trel], -2 );
    res(end+1,1) = scr(1,1);
end