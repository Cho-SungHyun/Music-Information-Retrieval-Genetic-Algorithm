function acc = bayes_matlab( train_x, train_y, test_x, test_y )

[~,~,train_y] = unique( train_y, 'rows' );
[~,~,test_y] = unique( test_y, 'rows' );

model = fitcnb( train_x, train_y,  'DistributionNames', 'mvmn' );

[pre,~] = predict( model, test_x  );
pre(isnan(pre),1) = 0;
acc = sum(pre==test_y) / size(test_y,1);

end