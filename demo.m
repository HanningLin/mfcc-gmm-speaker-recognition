 
clc,clear

waveDir='trainning\';

speakerData = dir(waveDir);
speakerData(1:2) = [];
speakerNum=length(speakerData);  % 人数；

% ======  特征提取

fprintf('\n 读取语音文件并进行特征提取...\n\n');
for i=1:speakerNum
	fprintf('\n第%d个人特征提取 ', i );
    [y, fs]=audioread(['trainning\' speakerData(i,1).name]);
    y=double(y);
    y=y/max(y);
    % 归一化处理
    %去掉杂音
    % 端点检测
    epInSampleIndex = epdByVol(y, fs);		 
    y=y(epInSampleIndex(1):epInSampleIndex(2)); 
 % 读入每个语音并求出 mfcc
    speakerData(i).mfcc=melcepst(y,8000); 

 
end
  
save speakerData  
fprintf('\n')
clear all;

 
% ====== GMM training

fprintf('\n训练每个语者的高斯混合模型...\n\n');

load speakerData.mat
%    每个GMM 中高斯模型的数量 ；10以上效果比较好
gaussianNum=10;		
speakerNum=length(speakerData);

for i=1:speakerNum
	fprintf('\n为第%d个语者%s训练GMM……', i,speakerData(i).name(1:end-4));
	%  对每个人说话的语音训练高斯模型，在识别的时候进行和每个测试数据的mfcc 进行比较
    [speakerGmm(i).mu, speakerGmm(i).sigm,speakerGmm(i).c] = gmm_estimate(speakerData(i).mfcc(:,5:12)',gaussianNum,20);
    % 用mfcc 5~12 训练出来高斯模型中的 三个参数
    fprintf(' end ');
end

fprintf('\n');
save speakerGmm ;
clear all;


% ====== recognition 开始识别

fprintf('\n开始识别...\n\n');
load speakerData;
load speakerGmm;

waveDir='testing\';
Test_speakerData = dir(waveDir);
Test_speakerData(1:2) = [];
Test_speakerNum=length(Test_speakerData);

count=0;
for i=1:Test_speakerNum
    %  
 [testing_data, fs]=audioread(['testing\' Test_speakerData(i,1).name]);
 % 求出测试集和每个训练集和mfcc系数比较
match= MFCC_feature_compare(testing_data,speakerGmm);
[max_1, index]=max(match); 
% 以最大值表示match的mfcc系数
fprintf('\n %s',max_1);
fprintf('\n所识别的说话人是%s\n。',speakerData(index).name(1:end-4))
if index==i   % 判断识别出的系数  如果识别出来，识别准确率count+1
    count=count+1;
end   
end

% ====== 计算准确率
count=double(count/i);
 fprintf('\n\n正确率为 %d',count);

