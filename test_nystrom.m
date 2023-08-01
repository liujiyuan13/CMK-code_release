clear
clc

save_path = './save/';
data_path = './data/';
data_name = 'Winnipeg1_fea';
norm = 'True';
dim = 128;
batch_size = 1024;
lr = 1;
epochs = 150;
kerneltype = 'Gaussian';

load([save_path, data_name, '/norm_', norm, '/dim_', num2str(dim), '/batch_size_', num2str(batch_size), '/lr_', num2str(lr), '/epochs_', num2str(epochs), '/', kerneltype, '_cmk.mat'], 'weights');
load([data_path, data_name, '.mat'], 'X', 'Y');
gt = double(squeeze(Y));
gt = gt - min(gt) + 1;
V = length(X);
N = size(X{1}, 2);

for v=1:V
    X{v} = X{v}'*double(weights{v})';
    X{v} = X{v} ./ (sqrt(sum(X{v}.^2, 2))+eps);
end

for v=1:V
    Xdic{v} = X{v}(ind_dic,:);
end

if ik == 1
    options.KernelType = 'Gaussian';
    options.t = 1.0;
elseif ik == 2
    options.KernelType = 'Linear';
elseif ik==3
    options.KernelType = 'PolyPlus';
    options.d = 2.0;
elseif ik == 4
    options.KernelType = 'Sigmoid';
    options.c = 0;
    options.d = 2;
elseif ik == 5
    options.KernelType = 'Cauchy';
    options.sigma = 1;
end

% kkm 
Ks = zeros(N, num_dic, V);
Kdics = zeros(num_dic, num_dic, V);
for v = 1:V
    Ks(:,:,v) = construct_kernel(X{v}, Xdic{v}, options);
    Kdics(:,:,v) = construct_kernel(Xdic{v}, Xdic{v}, options);
    res_kkm_contrastive(:,v,ik) = val_kernel_kkm_nystrom(Ks(:,:,v), Kdics(:,:,v), gt);
end
res_kkm_contrastive(:,v+1,ik) = val_kernel_kkm_nystrom(mean(Ks,3), mean(Kdics,3), gt);
