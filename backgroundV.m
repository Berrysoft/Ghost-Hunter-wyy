function [backg] = backgroundV(wave)
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
wave(wave<970)=[];
backg=mean(wave);
end

