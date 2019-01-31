function J = costFunctionJ(X, y, theta) % 函数名需要和文件名一致
m = size(X, 1);   % number of training examples, size of rows
predictions = X * theta; % predictions of hapothesis on all m examples
sqrErrors = (predictions - y) .^ 2; % squared errors .^ 指的是对数据中每个元素平方
J = 1 / (2 * m) * sum(sqrErrors); 