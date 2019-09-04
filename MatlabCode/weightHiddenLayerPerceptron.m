function randomWeightIneachHiddenLayerPerceptron= weightHiddenLayerPerceptron(RowIndex,ColIndex,theNumOfHiddenLayer)
    totalSize=RowIndex*ColIndex;
    tempW=[];
    tempw=[];
    
    %xmin=0;
    %xmax=0.1;
    
    rate=0.1;
    
    xmin=(-6/totalSize)*rate;
    xmax=(6/totalSize)*rate;
    
    
    %xmin=0.01;
    %xmax=0.02;
    
    %xmin=0.001;
    %xmax=0.002;
    
    %xmin=-1000;
    %xmax=1000;
    
    n=totalSize;
    for i=1:theNumOfHiddenLayer
        tempw=xmin+rand(1,n)*(xmax-xmin);
        tempW=[tempW;tempw];
    end
    randomWeightIneachHiddenLayerPerceptron=tempW;
    
end