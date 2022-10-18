function [pre,post] = mlnb( train, answer, test )

% Multi Label Naive Bayes
lcol = size( answer, 2 );
pre = zeros( size(test,1), lcol );
post = zeros( size(test,1), lcol );

% fprintf( 'Done: 0.0000' );
for k=1:lcol
    model = fitcnb( train, answer(:,k), 'dist', 'mvmn' );

    [pre(:,k), t] = model.predict(test);
    t(isnan(t(:,end)),end) = 0;
    pre(isnan(pre(:,k)),k) = 0;

    post(:,k) = t(:,end);
%     fprintf( '\b\b\b\b\b\b' );
%     fprintf( '%1.4f', k/lcol );    
end
% fprintf( '\n' );
