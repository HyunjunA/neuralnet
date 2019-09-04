function xMatrixS=calculateXfromS(sMatrixS)
    sMatrixLength=size(sMatrixS,1);
    xTemp=[];
    x=[];
    for id=1:sMatrixLength
        x=1/(1+exp(-1*sMatrixS(id)));
        xTemp=[xTemp;x];
    end
    xMatrixS=xTemp;
    
end
