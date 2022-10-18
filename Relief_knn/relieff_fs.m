function res = relieff_fs(data, answer, numFeat, k)

[~,~,answer] = unique( answer, 'rows' );

[r, ~] = relieff(data, answer, k);

res = r(1:numFeat);

end

