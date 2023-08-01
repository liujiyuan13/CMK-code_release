function [indx] = my_kmeans(U, numclass)

U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,size(U,2));
indx = kmeans(U_normalized,numclass, 'MaxIter',100, 'Replicates',50, 'EmptyAction','drop');
indx = indx(:);
  
end