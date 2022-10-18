function acc = knn_matlab( train_x, train_y, test_x, test_y , N_neighbors)

[~,~,train_y] = unique( train_y, 'rows' );
[~,~,test_y] = unique( test_y, 'rows' );


model = fitcknn( train_x, train_y, 'NumNeighbors', N_neighbors, 'Distance', 'cosine','Standardize', 1 );

[pre,~] = predict( model, test_x  );
pre(isnan(pre),1) = 0;
acc = sum(pre==test_y) / size(test_y,1);

end


