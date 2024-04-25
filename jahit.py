import cv2
import glob

image_paths=glob.glob('AmbilData9/*.jpg')
# image_paths1=glob.glob('DataKamera2/2/*.jpg')
# initialized a list of images
imgs = []
# imgs1 = []

for i in range(len(image_paths)):
	imgs.append(cv2.imread(image_paths[i]))
	imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.9,fy=0.9)
# for i in range(len(image_paths1)):
# 	imgs1.append(cv2.imread(image_paths1[i]))
# 	imgs1[i]=cv2.resize(imgs1[i],(0,0),fx=0.8,fy=0.8)

# cv2.imshow('1',imgs[0])

stitchy=cv2.Stitcher.create()
(dummy,output1)=stitchy.stitch(imgs)
cv2.imwrite('finalresult2.jpg',output1)
# (dummy,output2)=stitchy.stitch(imgs1)
# cv2.imwrite('op2.jpg',output2)
# (dummy,output)=stitchy.stitch(output1,output2)
# cv2.imwrite('op.jpg',output)

if dummy != cv2.STITCHER_OK:
	print("stitching ain't successful")
else:
	print('Your Panorama is ready!!!')

# final output
# cv2.imwrite('final result 2d.jpg',output)

cv2.waitKey(0)
