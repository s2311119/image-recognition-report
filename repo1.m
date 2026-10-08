function repo1()
% repo1
% Web画像を用いた2クラス画像分類の比較実験を行う。
%
% 以下の特徴抽出・分類手法を5-fold cross validationで評価する。
% - Color Histogram + Nearest Neighbor
% - Bag of Features (BoF) + SVM
% - BoF + Explicit Feature Map + Linear SVM
% - Pretrained CNN features + SVM
%
% CNNにはAlexNet, VGG16, ResNet101, DenseNet201を使用する。
    baseDir = fileparts(mfilename('fullpath'));

    posEasyFolderPaths = fullfile(baseDir, 'posImgDir_easy');
    posDiffFolderPaths = fullfile(baseDir, 'posImgDir_diff');
    negEasyFolderPaths = fullfile(baseDir, 'negImgDir_easy');
    negDiffFolderPaths = fullfile(baseDir, 'negImgDir_diff');
    posEasyDs = imageDatastore(posEasyFolderPaths, 'FileExtensions', '.jpg');
    posDiffDs = imageDatastore(posDiffFolderPaths, 'FileExtensions', '.jpg');
    negEasyDs = imageDatastore(negEasyFolderPaths, 'FileExtensions', '.jpg');
    negDiffDs = imageDatastore(negDiffFolderPaths, 'FileExtensions', '.jpg');
    poseasylist = posEasyDs.Files;
    posdifflist = posDiffDs.Files;
    negeasylist = negEasyDs.Files;
    negdifflist = negDiffDs.Files;
    rng(1);
    allEasyList = [poseasylist; negeasylist];
    allDiffList = [posdifflist; negdifflist];
    easy_labels = [ones(numel(poseasylist), 1); zeros(numel(negeasylist), 1)];
    diff_labels = [ones(numel(posdifflist), 1); zeros(numel(negdifflist), 1)];
    numEasyImages_pos = length(poseasylist);
    numDiffImages_pos = length(posdifflist);
    numEasyImages_neg = length(negeasylist);
    numDiffImages_neg = length(negdifflist);
    numEasy_Images = numEasyImages_pos + numEasyImages_neg;
    numDiff_Images = numDiffImages_pos + numDiffImages_neg;
    % (1) カラーヒストグラム + 最近傍
    % カラーヒストグラム(特徴抽出)
    hists_e = zeros(numEasy_Images, 64);
    hists_d = zeros(numDiff_Images, 64);
    for i = 1 : numEasy_Images
        img = imread(allEasyList{i});
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end
        feat = floor(double(img(:,:,1)) / 64) * 4 * 4 ...
            + floor(double(img(:, :, 2)) / 64) * 4 ...
            + floor(double(img(:, :, 3)) / 64);
        h = histcounts(feat(:), 0:64);
        hists_e(i, :) = h / sum(h);
    end
    for i = 1 : numDiff_Images
        img = imread(allDiffList{i});
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end
        feat = floor(double(img(:,:,1)) / 64) * 4 * 4 ...
            + floor(double(img(:, :, 2)) / 64) * 4 ...
            + floor(double(img(:, :, 3)) / 64);
        h = histcounts(feat(:), 0:64);
        hists_d(i, :) = h / sum(h);
    end
    % カラーヒストグラム(交差検証)
    k = 5;
    cv_e = cvpartition(easy_labels, 'KFold', k);
    cv_d = cvpartition(diff_labels, 'KFold', k);
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = hists_e(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = hists_e(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcknn(XTrain, YTrain, 'NumNeighbors', 1);
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (Color Hist + KNN, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');

    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = hists_d(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = hists_d(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcknn(XTrain, YTrain, 'NumNeighbors', 1);
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (Color Hist + KNN, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');

    % (2) BoF + SVM(非線形)
    if exist('bof_easy.mat', 'file') && exist('bof_diff.mat', 'file')
        fprintf('保存済みのBoFデータを読み込みます...\n');
        load('bof_easy.mat', 'bof_easy');
        load('bof_diff.mat', 'bof_diff');
    else
        fprintf('BoFデータの新規作成を開始します（時間がかかります）...\n');
        makeCodebookBoF(poseasylist, posdifflist, negeasylist, negdifflist);
        load('bof_easy.mat', 'bof_easy');
        load('bof_diff.mat', 'bof_diff');
    end

    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = bof_easy(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = bof_easy(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (BoF + 非線形SVM, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');

    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = bof_diff(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = bof_diff(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (BoF + 非線形SVM, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');

    % (3) DCNN特徴量 + 線形SVM
    net1 = alexnet;
    net2 = vgg16;
    IM1_e = [];
    IM1_d = [];
    IM2_e = [];
    IM2_d = [];
    for i = 1:numEasy_Images
        img = imread(allEasyList{i});
        reimg1 = imresize(img, net1.Layers(1).InputSize(1:2));
        reimg2 = imresize(img, net2.Layers(1).InputSize(1:2));
        IM1_e = cat(4, IM1_e, reimg1);
        IM2_e = cat(4, IM2_e, reimg2);
    end
    for i = 1:numDiff_Images
        img = imread(allDiffList{i});
        reimg1 = imresize(img, net1.Layers(1).InputSize(1:2));
        reimg2 = imresize(img, net2.Layers(1).InputSize(1:2));
        IM1_d = cat(4, IM1_d, reimg1);
        IM2_d = cat(4, IM2_d, reimg2);
    end
    dcnnf1_e = squeeze(activations(net1, IM1_e, 'fc7'))';
    dcnnf2_e = squeeze(activations(net2, IM2_e, 'fc7'))';
    dcnnf1_d = squeeze(activations(net1, IM1_d, 'fc7'))';
    dcnnf2_d = squeeze(activations(net2, IM2_d, 'fc7'))';
    vec_lengths1_e = sqrt(sum(dcnnf1_e.^2, 2));
    vec_lengths2_e = sqrt(sum(dcnnf2_e.^2, 2));
    vec_lengths1_d = sqrt(sum(dcnnf1_d.^2, 2));
    vec_lengths2_d = sqrt(sum(dcnnf2_d.^2, 2));
    dcnnf1_e = dcnnf1_e ./ vec_lengths1_e;
    dcnnf2_e = dcnnf2_e ./ vec_lengths2_e;
    dcnnf1_d = dcnnf1_d ./ vec_lengths1_d;
    dcnnf2_d = dcnnf2_d ./ vec_lengths2_d;
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = dcnnf1_e(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = dcnnf1_e(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (alexnetのDCNN特徴量 + 線形SVM, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = dcnnf1_d(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = dcnnf1_d(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (alexnetのDCNN特徴量 + 線形SVM, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = dcnnf2_e(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = dcnnf2_e(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (vgg16のDCNN特徴量 + 線形SVM, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = dcnnf2_d(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = dcnnf2_d(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (vgg16のDCNN特徴量 + 線形SVM, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    
    % アドバンストな内容
    fprintf('ここからはアドバンストな内容です。\n');

    % (4) BOF+線形SVM+feature maps
    data_easy = bof_easy;
    data3_easy = repmat(sqrt(abs(data_easy)).*sign(data_easy),[1 3]).*[0.8*ones(size(data_easy)) 0.6*cos(0.6*log(abs(data_easy)+eps)) 0.6*sin(0.6*log(abs(data_easy)+eps))];
    data_diff = bof_diff;
    data3_diff = repmat(sqrt(abs(data_diff)).*sign(data_diff),[1 3]).*[0.8*ones(size(data_diff)) 0.6*cos(0.6*log(abs(data_diff)+eps)) 0.6*sin(0.6*log(abs(data_diff)+eps))];
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = data3_easy(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = data3_easy(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (BOF+線形SVM+feature maps, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = data3_diff(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = data3_diff(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (BOF+線形SVM+feature maps, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');

    % (5) ResNet, DenseNetによるDCNN特徴量 + 線形SVM
    net1 = resnet101;
    net2 = densenet201;
    IM1_e = [];
    IM1_d = [];
    IM2_e = [];
    IM2_d = [];
    for i = 1:numEasy_Images
        img = imread(allEasyList{i});
        reimg1 = imresize(img, net1.Layers(1).InputSize(1:2));
        reimg2 = imresize(img, net2.Layers(1).InputSize(1:2));
        IM1_e = cat(4, IM1_e, reimg1);
        IM2_e = cat(4, IM2_e, reimg2);
    end
    for i = 1:numDiff_Images
        img = imread(allDiffList{i});
        reimg1 = imresize(img, net1.Layers(1).InputSize(1:2));
        reimg2 = imresize(img, net2.Layers(1).InputSize(1:2));
        IM1_d = cat(4, IM1_d, reimg1);
        IM2_d = cat(4, IM2_d, reimg2);
    end
    dcnnf1_e = squeeze(activations(net1, IM1_e, 'pool5'))';
    dcnnf2_e = squeeze(activations(net2, IM2_e, 'avg_pool'))';
    dcnnf1_d = squeeze(activations(net1, IM1_d, 'pool5'))';
    dcnnf2_d = squeeze(activations(net2, IM2_d, 'avg_pool'))';
    vec_lengths1_e = sqrt(sum(dcnnf1_e.^2, 2));
    vec_lengths2_e = sqrt(sum(dcnnf2_e.^2, 2));
    vec_lengths1_d = sqrt(sum(dcnnf1_d.^2, 2));
    vec_lengths2_d = sqrt(sum(dcnnf2_d.^2, 2));
    dcnnf1_e = dcnnf1_e ./ vec_lengths1_e;
    dcnnf2_e = dcnnf2_e ./ vec_lengths2_e;
    dcnnf1_d = dcnnf1_d ./ vec_lengths1_d;
    dcnnf2_d = dcnnf2_d ./ vec_lengths2_d;
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = dcnnf1_e(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = dcnnf1_e(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (resnet101のDCNN特徴量 + 線形SVM, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = dcnnf1_d(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = dcnnf1_d(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (resnet101のDCNN特徴量 + 線形SVM, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = dcnnf2_e(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = dcnnf2_e(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (densenet201のDCNN特徴量 + 線形SVM, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = dcnnf2_d(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = dcnnf2_d(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (densenet201のDCNN特徴量 + 線形SVM, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    % (6) Alexnet, VGG16, ResNet, DenseNetによるDCNN特徴量 + 非線形SVM
    net1 = alexnet;
    net2 = vgg16;
    net3 = resnet101;
    net4 = densenet201;
    IM1_e = [];
    IM1_d = [];
    IM2_e = [];
    IM2_d = [];
    IM3_e = [];
    IM3_d = [];
    IM4_e = [];
    IM4_d = [];
    for i = 1:numEasy_Images
        img = imread(allEasyList{i});
        reimg1 = imresize(img, net1.Layers(1).InputSize(1:2));
        reimg2 = imresize(img, net2.Layers(1).InputSize(1:2));
        reimg3 = imresize(img, net3.Layers(1).InputSize(1:2));
        reimg4 = imresize(img, net4.Layers(1).InputSize(1:2));
        IM1_e = cat(4, IM1_e, reimg1);
        IM2_e = cat(4, IM2_e, reimg2);
        IM3_e = cat(4, IM3_e, reimg3);
        IM4_e = cat(4, IM4_e, reimg4);
    end
    for i = 1:numDiff_Images
        img = imread(allDiffList{i});
        reimg1 = imresize(img, net1.Layers(1).InputSize(1:2));
        reimg2 = imresize(img, net2.Layers(1).InputSize(1:2));
        reimg3 = imresize(img, net3.Layers(1).InputSize(1:2));
        reimg4 = imresize(img, net4.Layers(1).InputSize(1:2));
        IM1_d = cat(4, IM1_d, reimg1);
        IM2_d = cat(4, IM2_d, reimg2);
        IM3_d = cat(4, IM3_d, reimg3);
        IM4_d = cat(4, IM4_d, reimg4);
    end
    dcnnf1_e = squeeze(activations(net1, IM1_e, 'fc7'))';
    dcnnf2_e = squeeze(activations(net2, IM2_e, 'fc7'))';
    dcnnf3_e = squeeze(activations(net3, IM3_e, 'pool5'))';
    dcnnf4_e = squeeze(activations(net4, IM4_e, 'avg_pool'))';
    dcnnf1_d = squeeze(activations(net1, IM1_d, 'fc7'))';
    dcnnf2_d = squeeze(activations(net2, IM2_d, 'fc7'))';
    dcnnf3_d = squeeze(activations(net3, IM3_d, 'pool5'))';
    dcnnf4_d = squeeze(activations(net4, IM4_d, 'avg_pool'))';
    vec_lengths1_e = sqrt(sum(dcnnf1_e.^2, 2));
    vec_lengths2_e = sqrt(sum(dcnnf2_e.^2, 2));
    vec_lengths3_e = sqrt(sum(dcnnf3_e.^2, 2));
    vec_lengths4_e = sqrt(sum(dcnnf4_e.^2, 2));
    vec_lengths1_d = sqrt(sum(dcnnf1_d.^2, 2));
    vec_lengths2_d = sqrt(sum(dcnnf2_d.^2, 2));
    vec_lengths3_d = sqrt(sum(dcnnf3_d.^2, 2));
    vec_lengths4_d = sqrt(sum(dcnnf4_d.^2, 2));
    dcnnf1_e = dcnnf1_e ./ vec_lengths1_e;
    dcnnf2_e = dcnnf2_e ./ vec_lengths2_e;
    dcnnf3_e = dcnnf3_e ./ vec_lengths3_e;
    dcnnf4_e = dcnnf4_e ./ vec_lengths4_e;
    dcnnf1_d = dcnnf1_d ./ vec_lengths1_d;
    dcnnf2_d = dcnnf2_d ./ vec_lengths2_d;
    dcnnf3_d = dcnnf3_d ./ vec_lengths3_d;
    dcnnf4_d = dcnnf4_d ./ vec_lengths4_d;
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = dcnnf1_e(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = dcnnf1_e(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (alexnetのDCNN特徴量 + 非線形SVM, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = dcnnf1_d(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = dcnnf1_d(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (alexnetのDCNN特徴量 + 非線形SVM, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = dcnnf2_e(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = dcnnf2_e(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (vgg16のDCNN特徴量 + 非線形SVM, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = dcnnf2_d(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = dcnnf2_d(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (vgg16のDCNN特徴量 + 非線形SVM, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');

    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = dcnnf3_e(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = dcnnf3_e(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (resnet101のDCNN特徴量 + 非線形SVM, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = dcnnf3_d(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = dcnnf3_d(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (resnet101のDCNN特徴量 + 非線形SVM, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_e, i);
        testIdx = test(cv_e, i);
        XTrain = dcnnf4_e(trainIdx, :);
        YTrain = easy_labels(trainIdx);
        XTest = dcnnf4_e(testIdx, :);
        YTest = easy_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allEasyList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (densenet201のDCNN特徴量 + 非線形SVM, easy): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
    accuracies = zeros(k, 1);
    for i = 1:k
        fprintf('Fold %d/%d... ', i, k);
        trainIdx = training(cv_d, i);
        testIdx = test(cv_d, i);
        XTrain = dcnnf4_d(trainIdx, :);
        YTrain = diff_labels(trainIdx);
        XTest = dcnnf4_d(testIdx, :);
        YTest = diff_labels(testIdx);
        mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        YPred = predict(mdl, XTest);
        acc = mean(YPred == YTest);
        % ミスった, 正解画像の出力
        testFiles = allDiffList(testIdx);
        tpIdx = find(YTest == YPred);
        wrongIdx = find(YPred ~= YTest);
        if ~isempty(tpIdx)
            fprintf('  [正解画像]:\n');
            for w = 1:min(3, length(tpIdx)) 
                idx = tpIdx(w);
                fprintf('    %s\n', testFiles{idx});
            end
        end
        if ~isempty(wrongIdx)
            fprintf('  [不正解画像]:\n');
            for w = 1:min(5, length(wrongIdx))
                idx = wrongIdx(w);
                fprintf('    正解:%d, 予測:%d -> %s\n', YTest(idx), YPred(idx), testFiles{idx});
            end
        end
        % ---ここまで---
        accuracies(i) = acc;
        fprintf('Accuracy: %.2f%%\n', acc * 100);
    end
    fprintf('--------------------------------\n');
    fprintf('平均分類精度 (densenet201のDCNN特徴量 + 非線形SVM, difficult): %.2f%%\n', mean(accuracies) * 100);
    fprintf('--------------------------------\n');
end