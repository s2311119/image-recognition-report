function repo2()
% repo2
% Flickr画像検索結果の再ランキング実験を行う。
%
% Flickrから取得した画像からVGG16の特徴量を抽出し、
% 線形SVMを学習する。
% ノイズを含むテスト画像に対してSVMスコアを計算し、
% スコアの高い順に並べ替えることで検索結果を再ランキングする。
    baseDir = fileparts(mfilename('fullpath'));
    % n = 25, キーワード"ramen", pos
    list1 = textread(fullfile(baseDir, 'pos_urllist_25_1.txt'), '%s');
    % n = 50, キーワード"ramen", pos
    list2 = textread(fullfile(baseDir, 'pos_urllist_50_1.txt'), '%s');
    % n = 25, キーワード"sushi", pos
    list3 = textread(fullfile(baseDir, 'pos_urllist_25_2.txt'), '%s');
    % n = 50, キーワード"sushi", pos
    list4 = textread(fullfile(baseDir, 'pos_urllist_50_2.txt'), '%s');
    OUTDIR1 = 'posImgDir_ramen_25';
    OUTDIR2 = 'posImgDir_ramen_50';
    OUTDIR3 = 'posImgDir_sushi_25';
    OUTDIR4 = 'posImgDir_sushi_50';
    
    if ~exist(OUTDIR1, 'dir')
        mkdir(OUTDIR1);
        for i = 1:size(list1, 1)
            fname = strcat(OUTDIR1,'/', num2str(i,'%04d'), '.jpg');
            websave(fname, list1{i});
        end
    end
    if ~exist(OUTDIR2, 'dir')
        mkdir(OUTDIR2);
        for i = 1:size(list2, 1)
            fname = strcat(OUTDIR2,'/', num2str(i,'%04d'), '.jpg');
            websave(fname, list2{i});
        end
    end
    if ~exist(OUTDIR3, 'dir')
        mkdir(OUTDIR3);
        for i = 1:size(list3, 1)
            fname = strcat(OUTDIR3,'/', num2str(i,'%04d'), '.jpg');
            websave(fname, list3{i});
        end
    end
    if ~exist(OUTDIR4, 'dir')
        mkdir(OUTDIR4);
        for i = 1:size(list4, 1)
            fname = strcat(OUTDIR4,'/', num2str(i,'%04d'), '.jpg');
            websave(fname, list4{i});
        end
    end
    % neg
    OUTDIR1 = 'negImgDir_750';
    if ~exist(OUTDIR1, 'dir')
        % path = fullfile('/MATLAB Drive/最終レポート/bgimg');
        path = fullfile(baseDir, 'bgimg');
        ds = imageDatastore(path, 'FileExtensions', '.jpg');
        list = ds.Files;
        mkdir(OUTDIR1);
        NegIdx = [1:size(list, 1)];
        rng(1);
        NegList = list(NegIdx(randperm(length(NegIdx), 750)));
        for j = 1:length(NegList)
            fnameNeg = strcat(OUTDIR1, '/', num2str(j, '%04d'), '.jpg');
            copyfile(NegList{j}, fnameNeg);
        end
    end
    net = vgg16;
    pos1 = fullfile(baseDir, 'posImgDir_ramen_25');
    pos2 = fullfile(baseDir, 'posImgDir_ramen_50');
    pos3 = fullfile(baseDir, 'posImgDir_sushi_25');
    pos4 = fullfile(baseDir, 'posImgDir_sushi_50');
    neg = fullfile(baseDir, 'negImgDir_750');
    pos1Ds = imageDatastore(pos1, 'FileExtensions', '.jpg');
    pos2Ds = imageDatastore(pos2, 'FileExtensions', '.jpg');
    pos3Ds = imageDatastore(pos3, 'FileExtensions', '.jpg');
    pos4Ds = imageDatastore(pos4, 'FileExtensions', '.jpg');
    negDs = imageDatastore(neg, 'FileExtensions', '.jpg');
    pos1list = pos1Ds.Files;
    pos2list = pos2Ds.Files;
    pos3list = pos3Ds.Files;
    pos4list = pos4Ds.Files;
    neglist = negDs.Files;
    list1 = [pos1list; neglist];
    list2 = [pos2list; neglist];
    list3 = [pos3list; neglist];
    list4 = [pos4list; neglist];
    IM1 = []; IM2 = []; IM3 = []; IM4 = [];
    for i = 1 : length(list1)
        img1 = imread(list1{i});
        img3 = imread(list3{i});
        if size(img1, 3) == 1
            img1 = cat(3, img1, img1, img1);
        end
        if size(img3, 3) == 1
            img3 = cat(3, img3, img3, img3);
        end
        reimg1 = imresize(img1, net.Layers(1).InputSize(1:2));
        reimg3 = imresize(img3, net.Layers(1).InputSize(1:2));
        IM1 = cat(4, IM1, reimg1);
        IM3 = cat(4, IM3, reimg3);
    end
    for i = 1 : length(list2)
        img2 = imread(list2{i});
        img4 = imread(list4{i});
        if size(img2, 3) == 1
            img2 = cat(3, img2, img2, img2);
        end
        if size(img4, 3) == 1
            img4 = cat(3, img4, img4, img4);
        end
        reimg2 = imresize(img2, net.Layers(1).InputSize(1:2));
        reimg4 = imresize(img4, net.Layers(1).InputSize(1:2));
        IM2 = cat(4, IM2, reimg2);
        IM4 = cat(4, IM4, reimg4);
    end
    dcnnf1 = squeeze(activations(net, IM1, 'fc7'))';
    vec_lengths1 = sqrt(sum(dcnnf1.^2, 2));
    dcnnf1 = dcnnf1 ./ vec_lengths1;

    dcnnf2 = squeeze(activations(net, IM2, 'fc7'))';
    vec_lengths2 = sqrt(sum(dcnnf2.^2, 2));
    dcnnf2 = dcnnf2 ./ vec_lengths2;
    
    dcnnf3 = squeeze(activations(net, IM3, 'fc7'))';
    vec_lengths3 = sqrt(sum(dcnnf3.^2, 2));
    dcnnf3 = dcnnf3 ./ vec_lengths3;

    dcnnf4 = squeeze(activations(net, IM4, 'fc7'))';
    vec_lengths4 = sqrt(sum(dcnnf4.^2, 2));
    dcnnf4 = dcnnf4 ./ vec_lengths4;

    Y1 = [ones(25, 1); ones(750, 1) .* (-1)];
    Y2 = [ones(50, 1); ones(750, 1) .* (-1)];
    Y3 = [ones(25, 1); ones(750, 1) .* (-1)];
    Y4 = [ones(50, 1); ones(750, 1) .* (-1)];
    mdl1 = fitcsvm(dcnnf1, Y1, 'KernelFunction', 'linear');
    mdl2 = fitcsvm(dcnnf2, Y2, 'KernelFunction', 'linear');
    mdl3 = fitcsvm(dcnnf3, Y3, 'KernelFunction', 'linear');
    mdl4 = fitcsvm(dcnnf4, Y4, 'KernelFunction', 'linear');
    % ramenはinteresting, sushiはlatestで300やった
    OUTDIR1 = 'test_noisy_ramen';
    OUTDIR2 = 'test_noisy_sushi';
    if ~exist(OUTDIR1, 'dir')
        list = textread(fullfile(baseDir, 'test_noisy_ramen.txt'), '%s');
        mkdir(OUTDIR1);
        for i = 1:size(list, 1)
            fname = strcat(OUTDIR1,'/', num2str(i,'%04d'), '.jpg');
            websave(fname, list{i});
        end
    end
    if ~exist(OUTDIR2, 'dir')
        list = textread(fullfile(baseDir, 'test_noisy_sushi.txt'), '%s');
        mkdir(OUTDIR2);
        for i = 1:size(list, 1)
            fname = strcat(OUTDIR2,'/', num2str(i,'%04d'), '.jpg');
            websave(fname, list{i});
        end
    end
    test1_path = fullfile(baseDir, 'test_noisy_ramen');
    test1Ds = imageDatastore(test1_path, 'FileExtensions', '.jpg');
    test1 = test1Ds.Files;

    test2_path = fullfile(baseDir, 'test_noisy_sushi');
    test2Ds = imageDatastore(test2_path, 'FileExtensions', '.jpg');
    test2 = test2Ds.Files;
    IM1 = []; IM2 = [];
    for i = 1 : length(test1)
        img1 = imread(test1{i});
        img2 = imread(test2{i});
        if size(img1, 3) == 1
            img1 = cat(3, img1, img1, img1);
        end
        if size(img2, 3) == 1
            img2 = cat(3, img2, img2, img2);
        end
        reimg1 = imresize(img1, net.Layers(1).InputSize(1:2));
        reimg2 = imresize(img2, net.Layers(1).InputSize(1:2));
        IM1 = cat(4, IM1, reimg1);
        IM2 = cat(4, IM2, reimg2);
    end
    dcnnf_test1 = squeeze(activations(net, IM1, 'fc7'))';
    vec_lengths1 = sqrt(sum(dcnnf_test1.^2, 2));
    dcnnf_test1 = dcnnf_test1 ./ vec_lengths1;
    
    dcnnf_test2 = squeeze(activations(net, IM2, 'fc7'))';
    vec_lengths2 = sqrt(sum(dcnnf_test2.^2, 2));
    dcnnf_test2 = dcnnf_test2 ./ vec_lengths2;

    [label1, score1] = predict(mdl1, dcnnf_test1);
    [label2, score2] = predict(mdl2, dcnnf_test1);
    [label3, score3] = predict(mdl3, dcnnf_test2);
    [label4, score4] = predict(mdl4, dcnnf_test2);
    [sorted_score1, sorted_idx1] = sort(score1(:,2), 'descend');
    [sorted_score2, sorted_idx2] = sort(score2(:,2), 'descend');
    [sorted_score3, sorted_idx3] = sort(score3(:,2), 'descend');
    [sorted_score4, sorted_idx4] = sort(score4(:,2), 'descend');
    % ramen25枚
    fid = fopen('ramen_25.html', 'w');
    fprintf(fid, '<html><body><h2>ポジティブ画像25枚のラーメン画像リランキング結果</h2><table border="1">');
    fprintf(fid, '<tr><th>Rank</th><th>Score</th><th>Image</th></tr>');
    for i = 1 : 100
        path = test1{sorted_idx1(i)};
        fid_img = fopen(path, 'r');
        raw_data = fread(fid_img, '*uint8');
        fclose(fid_img);
        b64_str = matlab.net.base64encode(raw_data);
        fprintf(fid, '<tr><td>%d</td><td>%.4f</td>', i, sorted_score1(i));
        fprintf(fid, '<td><img src="data:image/jpeg;base64,%s" width="150"></td></tr>', b64_str);
    end
    fprintf(fid, '</table></body></html>');
    fclose(fid);
    fprintf('ramen_25.html を作成しました。ブラウザで開いて確認してください。\n');
    % ramen50枚
    fid = fopen('ramen_50.html', 'w');
    fprintf(fid, '<html><body><h2>ポジティブ画像50枚のラーメン画像リランキング結果</h2><table border="1">');
    fprintf(fid, '<tr><th>Rank</th><th>Score</th><th>Image</th></tr>');
    for i = 1 : 100
        path = test1{sorted_idx2(i)};
        fid_img = fopen(path, 'r');
        raw_data = fread(fid_img, '*uint8');
        fclose(fid_img);
        b64_str = matlab.net.base64encode(raw_data);
        fprintf(fid, '<tr><td>%d</td><td>%.4f</td>', i, sorted_score2(i));
        fprintf(fid, '<td><img src="data:image/jpeg;base64,%s" width="150"></td></tr>', b64_str);
    end
    fprintf(fid, '</table></body></html>');
    fclose(fid);
    fprintf('ramen_50.html を作成しました。ブラウザで開いて確認してください。\n');
    % sushi25枚
    fid = fopen('sushi_25.html', 'w');
    fprintf(fid, '<html><body><h2>ポジティブ画像25枚の寿司画像リランキング結果</h2><table border="1">');
    fprintf(fid, '<tr><th>Rank</th><th>Score</th><th>Image</th></tr>');
    for i = 1 : 100
        path = test2{sorted_idx3(i)};
        fid_img = fopen(path, 'r');
        raw_data = fread(fid_img, '*uint8');
        fclose(fid_img);
        b64_str = matlab.net.base64encode(raw_data);
        fprintf(fid, '<tr><td>%d</td><td>%.4f</td>', i, sorted_score3(i));
        fprintf(fid, '<td><img src="data:image/jpeg;base64,%s" width="150"></td></tr>', b64_str);
    end
    fprintf(fid, '</table></body></html>');
    fclose(fid);
    fprintf('sushi_25.html を作成しました。ブラウザで開いて確認してください。\n');
    % sushi50枚
    fid = fopen('sushi_50.html', 'w');
    fprintf(fid, '<html><body><h2>ポジティブ画像50枚の寿司画像リランキング結果</h2><table border="1">');
    fprintf(fid, '<tr><th>Rank</th><th>Score</th><th>Image</th></tr>');
    for i = 1 : 100
        path = test2{sorted_idx4(i)};
        fid_img = fopen(path, 'r');
        raw_data = fread(fid_img, '*uint8');
        fclose(fid_img);
        b64_str = matlab.net.base64encode(raw_data);
        fprintf(fid, '<tr><td>%d</td><td>%.4f</td>', i, sorted_score4(i));
        fprintf(fid, '<td><img src="data:image/jpeg;base64,%s" width="150"></td></tr>', b64_str);
    end
    fprintf(fid, '</table></body></html>');
    fclose(fid);
    fprintf('sushi_50.html を作成しました。ブラウザで開いて確認してください。\n');
end