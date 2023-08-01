clear
clc

save_path = './save/';
data_name = 'BBC2';
norm = 'True';
dim = 128;
lr = 1;
epoch = 300;
kerneltype = 'Gaussian';

% load cmk results
load([save_path, data_name, '/norm_', norm, '/dim_', num2str(dim), '/lr_', num2str(lr), '/epochs_', num2str(epoch),  '/', kerneltype, '_cmk.mat'], 'K', 'gt');
% % load cmkkm results
% load([save_path, data_name, '/norm_', norm, '/dim_', num2str(dim), '/lr_', num2str(lr), '/epochs_', num2str(epoch),  '/', kerneltype, '_cmkkm.mat'], 'K', 'gt');
gt = double(squeeze(gt'));
gt = gt - min(gt) + 1;
V = size(K,3);

for v = 1:V
    res_kkm(:,v,inorm,il, idim, iepoch, ik) = val_kernel_kkm(K(:,:,v), gt);
end
res_kkm(:,v+1,inorm,il, idim, iepoch, ik) = val_kernel_kkm(mean(K,3), gt);


