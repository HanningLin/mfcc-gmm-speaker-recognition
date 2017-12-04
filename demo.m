 
clc,clear

waveDir='trainning\';

speakerData = dir(waveDir);
speakerData(1:2) = [];
speakerNum=length(speakerData);  % ������

% ======  ������ȡ

fprintf('\n ��ȡ�����ļ�������������ȡ...\n\n');
for i=1:speakerNum
	fprintf('\n��%d����������ȡ ', i );
    [y, fs]=audioread(['trainning\' speakerData(i,1).name]);
    y=double(y);
    y=y/max(y);
    % ��һ������
    %ȥ������
    % �˵���
    epInSampleIndex = epdByVol(y, fs);		 
    y=y(epInSampleIndex(1):epInSampleIndex(2)); 
 % ����ÿ����������� mfcc
    speakerData(i).mfcc=melcepst(y,8000); 

 
end
  
save speakerData  
fprintf('\n')
clear all;

 
% ====== GMM training

fprintf('\nѵ��ÿ�����ߵĸ�˹���ģ��...\n\n');

load speakerData.mat
%    ÿ��GMM �и�˹ģ�͵����� ��10����Ч���ȽϺ�
gaussianNum=10;		
speakerNum=length(speakerData);

for i=1:speakerNum
	fprintf('\nΪ��%d������%sѵ��GMM����', i,speakerData(i).name(1:end-4));
	%  ��ÿ����˵��������ѵ����˹ģ�ͣ���ʶ���ʱ����к�ÿ���������ݵ�mfcc ���бȽ�
    [speakerGmm(i).mu, speakerGmm(i).sigm,speakerGmm(i).c] = gmm_estimate(speakerData(i).mfcc(:,5:12)',gaussianNum,20);
    % ��mfcc 5~12 ѵ��������˹ģ���е� ��������
    fprintf(' end ');
end

fprintf('\n');
save speakerGmm ;
clear all;


% ====== recognition ��ʼʶ��

fprintf('\n��ʼʶ��...\n\n');
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
 % ������Լ���ÿ��ѵ������mfccϵ���Ƚ�
match= MFCC_feature_compare(testing_data,speakerGmm);
[max_1, index]=max(match); 
% �����ֵ��ʾmatch��mfccϵ��
fprintf('\n %s',max_1);
fprintf('\n��ʶ���˵������%s\n��',speakerData(index).name(1:end-4))
if index==i   % �ж�ʶ�����ϵ��  ���ʶ�������ʶ��׼ȷ��count+1
    count=count+1;
end   
end

% ====== ����׼ȷ��
count=double(count/i);
 fprintf('\n\n��ȷ��Ϊ %d',count);

