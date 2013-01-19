function rbfTry()

    
    
    csvData = csvread('blood.csv');
    
    inputs = csvData(:, 1:4);
    outputs = csvData(:, 5);
    
    t = cputime;
    svm = svmtrain(inputs, outputs);
    
    testedGrps = svmclassify(svm, inputs);
    
    results = sum(abs(outputs - testedGrps)) / size(outputs,1)
  
    cputime - t
    
end