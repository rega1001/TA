ref = imread("finalresult.jpg");
a = imread("comb.jpg");
b = imread("hasilMatlab2citra.jpg");
%%
scale =730/1516;
ref1 = imresize(ref,scale);
scale2 =730/3648;
b = imresize(b,scale2);
%%
ref1 = ref1(:,1:1451,:);
a = a(:,1:1451,:);

%%
figure()
imshow(ref1);
figure()
imshow(a);
figure()
imshow(b);
%%
peaksnr = psnr(a,ref1)
peaksnr2 = psnr(b,ref1)
imwrite(b,"zm.jpg")
imwrite(a,"zp.jpg")