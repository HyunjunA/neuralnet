function EpsiS=backpropagation(epsW,epsX,epsS,epsY)
EpsiS=[];
tempEpsiLast=[];
tempEpsiFirst=[];%size should be 100*1.
temp=[];

tempEpsiLast=2*(epsX{3}-epsY)*(epsX{3})*(1-epsX{3});

for epsi=1:size(epsW{2},2)
    temp=tempEpsiLast*(epsW{2}(epsi))*(epsX{2}(epsi))*(1-(epsX{2}(epsi)));
    tempEpsiFirst=[tempEpsiFirst,temp];
end

EpsiS{1}=tempEpsiFirst;
EpsiS{2}=tempEpsiLast;



end