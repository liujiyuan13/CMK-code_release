function [eval] = val_kernel_kkm(K, gt)

gt = gt - min(gt) + 1;
num_class = length(unique(gt));

[H] = my_kernel_kmeans(K, num_class);
[y] = my_kmeans(H, num_class);
[eval] = my_eval_y(y, gt);


end