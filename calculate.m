thresh=9; %�趨��ֵ����������ֵ�ļ���һ��PE�ź�
PEtime = cell(len,1); %Ԥ����PE����ʱ����ڴ棬ÿ�����η���һ��cell����
WEIGHT = cell(len,1); %Ԥ����Ȩ�ص��ڴ棬ÿ�����η���һ��cell����

tic %��ʼ��ʱ
for i=1:len %�����в��δ����ѭ��
    thiswave = TestWave.Waveform(:,i)'; %�Ѳ��ζ�ȡ��1*1029��������������thiswave
    baseline = backgroundV(thiswave(1:200)); %ѡȡǰ200ns������baseline
    thiswave = int16(baseline) - thiswave; %��ȥbaseline
    
    if thiswave<thresh %���û���κβ��ι���ֵ����ѡȡ������ߵ���ΪPEtime��ͬʱȨ��Ϊ1
        tot=find(thiswave==max(thiswave)); %�ҵ�������ߵ�ʱ��Ϊtot
        finalpetime = tot-7; %�趨ƫ����Ϊ7
        weigh = ones(1,length(finalpetime)); %Ȩ��Ϊ1
    else %����й����ź�
        begining = thiswave(1:10)<thresh; %���ǰ10ns������û�й��У�û����ȫΪ1��
        if begining %ȫΪ1����û��
            tot=0; %��ʼ��tot
            finalpetime=[]; %��ʼ�����յõ���petime
            weigh=[]; %��ʼ��Ȩ��
        else
            tot=find(~begining,1); %���ǰ10ns���й����ź�
            finalpetime = tot-6;% �ҵ���������ƫ��
            weigh = 1;
            thiswave(1:10)=0; %Ȼ���ǰ10ns�ź�����Ϊ0�����������Ƿ�ֹ�а�����ο���ǰ10ns���³���bug��
        end
        while true %�����㷨���Ĳ��ֵ�ѭ������ȥÿ���ҵ��Ĳ���
            wave = thiswave; %�����������ڴ���
            wave(wave<thresh)=0; %δ���е���Ϊ����������Ϊ0
            tot = find(wave,1); %�ҵ���һ������ʱ��tot
            if isempty(tot) %�����Ҳ�Ҳ�����������ȫ���ҳ������ź�
                break %��˽���ѭ��
            end
            petime = tot-6; %����ҵ��ˣ���ô��ȥƫ�����õ�����źŵ�petime
            
            if isempty(finalpetime) || (petime-finalpetime(end))<3 || (petime-finalpetime(end))>5 || thiswave(peaktime)>12 %һЩcut��������Щ
                weigh = [weigh,1];
                finalpetime=[finalpetime,petime];
            end
            if length(finalpetime)>500 %��������˳�����500��petime��Ӧ������������ѭ����������ֹ������
                error('too many PEs found, there must be a bug.')
            end

            thiswave = thiswave - int16(modelfunc([petime+para(1),petime+para(2),para(3),para(4)],(1:1029))); %�㷨��꣺�ڲ����ϼ�ȥ�ѷ��ֵ��źŶ�Ӧ�ı�׼����Ӳ��Σ��������ǾͿ��Խ����������źŵ��ӵĲ�����ȥ��һ������Ӳ��Ρ�

        end %���������ź�+��ȥ�ѷ����źŵ�ѭ��
    end 
    PEtime{i}=finalpetime;%����һ�����εĴ���д������ʱ��
    WEIGHT{i}=weigh;%����һ�����εĴ���д��Ȩ��
end
toc %������ʱ
