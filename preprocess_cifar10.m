clear all; close all; clc;
load cifar-10/batches.meta.mat

%%
imgs = zeros(50000, 3072);
for i = 1:5 
    eval(['load cifar-10/data_batch_', num2str(i), '.mat']);
    for j = 1:10000
        img = permute(reshape(data(j, :), [32, 32, 3]), [2, 1, 3]);
        img_float = single(img) / 255;
        img_float2 = permute(img_float, [2, 1, 3]);
        imgs((i - 1) * 10000 + j, :) = img_float2(:)';
    end
end

imgs2 = zeros(10000, 3072);
eval('load cifar-10/test_batch.mat');
for j = 1:10000
    img = permute(reshape(data(j, :), [32, 32, 3]), [2, 1, 3]);
    img_float = single(img) / 255;
    img_float2 = permute(img_float, [2, 1, 3]);
    imgs2(j, :) = img_float2(:)';
end

save('cifar_10_train.mat', 'imgs');
save('cifar_10_test.mat', 'imgs2');