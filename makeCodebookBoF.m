function makeCodebookBoF(poseasylist, posdifflist, negeasylist, negdifflist)
% makeCodebookBoF
% Bag of Features (BoF)用のコードブックと特徴ベクトルを生成する。
%
% 各画像から局所特徴を抽出し、k-meansによってvisual wordsを作成する。
% 各画像をvisual wordの出現頻度ヒストグラムとして表現し、
% 分類用のBoF特徴量として保存する。
    baseDir = fileparts(mfilename('fullpath'));
    fprintf('--- コードブック作成を開始します ---\n');
    fprintf('画像リストを作成中...\n');
    easy_list = {poseasylist{:} negeasylist{:}};
    diff_list = {posdifflist{:} negdifflist{:}};
    % コードブックの作成
    fprintf('(posImgDir_easy, negImgDir_easy)の画像からSURF特徴を抽出中...\n');
    Features = [];
    for i = 1:numel(easy_list)
        I = imread(easy_list{i});
        if size(I, 3) == 3
            I = rgb2gray(I);
        end
        p = createRandomPoints(I, 500);
        [f, ~] = extractFeatures(I,p);
        Features=[Features; f];
    end
    numFeatures = size(Features, 1);
    fprintf('抽出された特徴量の総数: %d 個\n', numFeatures);

    if numFeatures > 50000
        fprintf('特徴量が多すぎるため 50000個 にランダムサンプリングします。\n');
        Features = Features(randperm(numFeatures, 50000), :);
    end
    fprintf('K-means法でコードブック(サイズ1000)を作成中...\n');
    [~, CODEBOOK_easy] = kmeans(Features, 1000);

    save(fullfile(baseDir, 'codebook_easy.mat'), 'CODEBOOK_easy');
    fprintf('保存完了: codebook_easy.mat (サイズ: %d x %d)\n', size(CODEBOOK_easy));

    fprintf('(posImgDir_diff, negImgDir_diff)の画像からSURF特徴を抽出中...\n');
    Features = [];
    for i = 1:numel(diff_list)
        I = imread(diff_list{i});
        if size(I, 3) == 3
            I = rgb2gray(I);
        end
        p = createRandomPoints(I, 500);
        [f, ~] = extractFeatures(I,p);
        Features=[Features; f];
    end
    numFeatures = size(Features, 1);
    fprintf('抽出された特徴量の総数: %d 個\n', numFeatures);

    if numFeatures > 50000
        fprintf('特徴量が多すぎるため 50000個 にランダムサンプリングします。\n');
        Features = Features(randperm(numFeatures, 50000), :);
    end
    fprintf('K-means法でコードブック(サイズ1000)を作成中...\n');
    [~, CODEBOOK_diff] = kmeans(Features, 1000);

    % save('codebook_diff.mat', 'CODEBOOK_diff');
    save(fullfile(baseDir, 'codebook_diff.mat'), 'CODEBOOK_diff');
    fprintf('保存完了: codebook_diff.mat (サイズ: %d x %d)\n', size(CODEBOOK_diff));
    
    % BoFベクトル化
    fprintf('BoFベクトルに変換中...\n');
    bof_easy = zeros(numel(easy_list), 1000);
    bof_diff = zeros(numel(diff_list), 1000);
    codebook_sq_e = sum(CODEBOOK_easy.^2, 2)';
    codebook_sq_d = sum(CODEBOOK_diff.^2, 2)';
    for i = 1:numel(easy_list)
        I = imread(easy_list{i});
        if size(I, 3) == 3, I = rgb2gray(I); end
        p = createRandomPoints(I, 2500);
        [features, ~] = extractFeatures(I, p);
        if isempty(features), continue; end
        feat_sq = sum(features.^2, 2);
        feat_cb = features * CODEBOOK_easy';
        distMat = feat_sq + codebook_sq_e - 2 * feat_cb;
        [~, min_idx] = min(distMat, [], 2);
        bof_easy(i, :) = histcounts(min_idx, 1:1001);
    end
    bof_sum = sum(bof_easy, 2); bof_sum(bof_sum==0) = 1;
    bof_easy = bof_easy ./ bof_sum;
    fprintf('BoF行列(easy)の作成完了。サイズ: %d x %d\n', size(bof_easy));
    % save('bof_easy.mat', 'bof_easy', 'easy_list');
    save(fullfile(baseDir, 'bof_easy.mat'), 'bof_easy', 'easy_list');
    
    for i = 1:numel(diff_list)
        I = imread(diff_list{i});
        if size(I, 3) == 3, I = rgb2gray(I); end
        p = createRandomPoints(I, 2500);
        [features, ~] = extractFeatures(I, p);
        if isempty(features), continue; end
        feat_sq = sum(features.^2, 2);
        feat_cb = features * CODEBOOK_diff';
        distMat = feat_sq + codebook_sq_d - 2 * feat_cb;
        [~, min_idx] = min(distMat, [], 2);
        bof_diff(i, :) = histcounts(min_idx, 1:1001);
    end
    bof_sum = sum(bof_diff, 2); bof_sum(bof_sum==0) = 1;
    bof_diff = bof_diff ./ bof_sum;
    fprintf('BoF行列(difficult)の作成完了。サイズ: %d x %d\n', size(bof_diff));
    % save('bof_diff.mat', 'bof_diff', 'diff_list');
    save(fullfile(baseDir, 'bof_diff.mat'), 'bof_diff', 'diff_list');
end

function PT=createRandomPoints(I,num)
  [sy sx]=size(I);
  sz=[sx sy];
  for i=1:num
    s=0;
    while s<1.6
      s=randn()*3+3;
    end
    p=ceil((sz-ceil(s)*2).*rand(1,2)+ceil(s));
    if i==1
      PT=[SURFPoints(p,'Scale',s)];
    else
      PT=[PT; SURFPoints(p,'Scale',s)];
    end
  end
end