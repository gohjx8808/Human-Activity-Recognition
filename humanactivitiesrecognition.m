load X_train.txt;
load X_test.txt;
load y_train.txt;
load y_test.txt;
net = patternnet(10);
y=full(ind2vec(y_train',6));
yr=full(ind2vec(y_test',6));
net.divideParam.trainRatio=70/100;
net.divideParam.valRatio=15/100;
net.divideParam.testRatio=15/100;
[net,tr]=train(net,X_train',y,'useGPU','yes');
figure(1)
plotperform(tr)
saveas(gcf,'performance.png')
w1=net.IW{1};
w2=net.LW{2};
b1=net.b{1};
b2=net.b{2};
trainy=net(X_train');
trainerrors=gsubtract(y,trainy);
performtrain=perform(net,y,trainy)
%testing model
testy=net(xr');
testerrors=gsubtract(yr,testy);
performtest=perform(net,yr,testy)
performance=crossentropy(net,yr,testy,{1},'regularization',0.1)
testIndices=vec2ind(testy);
figure(2)
plotconfusion(yr,testy)
h=gca;
h.XTickLabel={'LAYING', 'SITTING','STANDING','WALKING','WALKING DOWNSTAIRS','WALKING UPSTAIRS',''};
h.YTickLabel={'LAYING', 'SITTING','STANDING','WALKING','WALKING DOWNSTAIRS','WALKING UPSTAIRS',''};
saveas(gcf,'confusion.png')
[c,ic]=confusion(yr,testy);
fprintf('Percentage Correct Classification: %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification: %f%%\n', 100*c);
figure(3)
plotroc(yr,testy)
saveas(gcf,'roc.png')