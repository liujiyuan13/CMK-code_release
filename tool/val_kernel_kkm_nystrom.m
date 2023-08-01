function [eval] = val_kernel_kkm(K, K_dic, gt)

gt = gt - min(gt) + 1;
num_class = length(unique(gt));
num_smp = length(gt);

[U, D] = eig(K_dic);
D = real(diag(D));
D(D>0) = D(D>0).^-0.5;
D = diag(D);
H = D * U'*K';

[y] = my_kmeans(H', num_class);
[eval] = my_eval_y(y, gt);


end