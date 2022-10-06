clc;
clear all;
close all;
load 'Corrupted_EEG.mat';


[r c]=size(Data);
cancelled=zeros(r,c);
clean_data=zeros(32,3200);
%%%%Subtracting the average trend using Golay Filtering%%%%%%%%%
for ii=1:r

primary=Data(ii,:);
rcfs=sgolayfilt(primary,5,127);
refer=rcfs;
cancelled(ii,:)=primary-refer;
end
%%%%Threholding Wavelet Coefficients%%%%%%%%%
for jj=1:r

[c,l] = wavedec(cancelled(jj,:),4,'db2');
 approx = appcoef(c,l,'db2');
 [cd1,cd2,cd3,cd4] = detcoef(c,l,[1 2 3 4]);

md1=cd1;
md2=cd2;
md3=cd3;
md4=cd4;
approx1=approx;
 
t=std(cd3)*0.8;
for ii=1:length(cd1)
if(abs(cd1(1,ii))>=t)
    md1(1,ii)=sign(cd1(1,ii))*t;
elseif(abs(cd1(1,ii))<t)
     md1(1,ii)=sign(cd1(1,ii))*abs(cd1(1,ii));
end
end

for ii=1:length(cd2)
if(abs(cd2(1,ii))>=t)
    md2(1,ii)=sign(cd2(1,ii))*t;
elseif(abs(cd2(1,ii))<t)
     md2(1,ii)=sign(cd2(1,ii))*abs(cd2(1,ii));
end
end

for ii=1:length(cd3)
if(abs(cd3(1,ii))>=t)
    md3(1,ii)=sign(cd3(1,ii))*t;
elseif(abs(cd3(1,ii))<t)
     md3(1,ii)=sign(cd3(1,ii))*abs(cd3(1,ii));
end
end

for ii=1:length(cd4)
if(abs(cd4(1,ii))>=t)
    md4(1,ii)=sign(cd4(1,ii))*t;
elseif(abs(cd4(1,ii))<t)
     md4(1,ii)=sign(cd4(1,ii))*abs(cd4(1,ii));
end
end

for ii=1:length(approx)
if(abs(approx(1,ii))>=t)
     approx1(1,ii)=sign(approx(1,ii))*t;
elseif(abs(approx(1,ii))<t)
     approx1(1,ii)=sign(approx(1,ii))*abs(approx(1,ii));
end
end
clean=waverec([approx1,md4,md3,md2,md1],[length(approx1),length(md4),length(md3),length(md2),length(md1),length(cancelled)],'db2');
Clean_data(jj,:)=clean;
 
end

save('Cleaned_EEG.mat', 'Clean_data');


