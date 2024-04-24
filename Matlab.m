I = imread("AmbilData1\0323145137.jpg");
J = imread("AmbilData1\0323145138.jpg");
% I = imcrop(I,[1030 590 70 70]);
% Initialize features for I(1)
grayImage = im2gray(I);
grayImage1 = im2gray(J);
points = detectORBFeatures(grayImage);
points1 = detectORBFeatures(grayImage1);
[features, points] = extractFeatures(grayImage,points);
[features1, points1] = extractFeatures(grayImage1,points1);

figure(2)
imshow(grayImage)
hold on
plot(points,'ShowScale',false)
hold off
figure(3)
imshow(grayImage1)
hold on
plot(points1,'ShowScale',false)
hold off
