import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

def draw_match(img1, img2, corr1, corr2):

    corr1 = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], 1) for i in range(corr1.shape[0])]
    corr2 = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], 1) for i in range(corr2.shape[0])]

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]

    display = cv2.drawMatches(img1, corr1, img2, corr2, draw_matches, None,
                              matchColor=(0, 255, 0),
                              singlePointColor=(0, 0, 255),
                              flags=4
                              )
    return display

def LPM_cosF(neighborX, neighborY, lbd, vec, d2, tau, K):
    L = neighborX.shape[0]
    C = 0
    Km = np.array([K+2, K, K-2])
    M = len(Km)

    for KK in Km:
        neighborX = neighborX[:,1:KK+1]
        neighborY = neighborY[:,1:KK+1]

        ## This is a loop implementation for computing c1 and c2, much slower but more readable
        # ni = np.zeros((L,1))
        # c1 = np.zeros((L,1))
        # c2 = np.zeros((L,1))
        # for i in range(L):
        #     inters = np.intersect1d(neighborX[i,:], neighborY[i,:])
        #     ni[i] = len(inters)
        #     c1[i] = KK - ni[i]
        #     cos_sita = np.sum(vec[inters, :]*vec[i,:],axis=1)/np.sqrt(d2[inters]*d2[i]).reshape(ni[i].astype('int').item(), 1)
        #     ratio = np.minimum(d2[inters], d2[i])/np.maximum(d2[inters], d2[i])
        #     ratio = ratio.reshape(-1,1)
        #     label = cos_sita*ratio < tau
        #     c2[i] = np.sum(label.astype('float64'))

        neighborIndex = np.hstack((neighborX,neighborY))
        index = np.sort(neighborIndex,axis=1)
        temp1 = np.hstack((np.diff(index,axis = 1),np.ones((L,1))))
        temp2 = (temp1==0).astype('int')
        ni = np.sum(temp2,axis=1)
        c1 = KK - ni
        temp3 = np.tile(vec.reshape((vec.shape[0],1,vec.shape[1])),(1,index.shape[1],1))*vec[index, :]
        temp4 = np.tile(d2.reshape((d2.shape[0],1)),(1,index.shape[1]))
        temp5 = d2[index]*temp4
        cos_sita = np.sum(temp3,axis=2).reshape((temp3.shape[0],temp3.shape[1]))/np.sqrt(temp5)
        ratio = np.minimum(d2[index], temp4)/np.maximum(d2[index], temp4)
        label = cos_sita*ratio < tau
        label = label.astype('int')
        c2 = np.sum(label*temp2,axis=1)

        C = C + (c1 + c2)/KK


    idx = np.where((C/M) <= lbd)
    return idx[0], C

def LPM_filter(X, Y):
    lambda1 = 0.8   
    lambda2 = 0.5  
    numNeigh1 = 6    
    numNeigh2 = 6
    tau1 = 0.2  
    tau2 = 0.2

    vec = Y - X
    d2 = np.sum(vec**2,axis=1)

    treeX = KDTree(X)
    _, neighborX = treeX.query(X, k=numNeigh1+3)
    treeY = KDTree(Y)
    _, neighborY = treeY.query(Y, k=numNeigh1+3)

    idx, C = LPM_cosF(neighborX, neighborY, lambda1, vec, d2, tau1, numNeigh1)

    if len(idx) >= numNeigh2 + 4:
        treeX2 = KDTree(X[idx,:])
        _, neighborX2 = treeX2.query(X, k=numNeigh2+3)
        treeY2 = KDTree(Y[idx,:])
        _, neighborY2 = treeY2.query(Y, k=numNeigh2+3)
        neighborX2 = idx[neighborX2]
        neighborY2 = idx[neighborY2]
        idx, C = LPM_cosF(neighborX2, neighborY2, lambda2, vec, d2, tau2, numNeigh2)

    mask = np.zeros((X.shape[0],1))
    mask[idx] = 1

    return mask.flatten().astype('bool')


def plot_images(*imgs, figsize=(30,20), hide_ticks=False):
    f = plt.figure(figsize=figsize)
    width = np.ceil(np.sqrt(len(imgs)))
    height = np.ceil(len(imgs) / width)
    for i, img in enumerate(imgs, 1):
        ax = f.add_subplot(int(height), int(width), i)
        if hide_ticks:
            ax.axis('off')
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

left = cv2.imread('Data Kamera\SAM_0042.jpg')
right = cv2.imread('Data Kamera\SAM_0043.jpg')

left = cv2.resize(left,None,fx=0.2,fy=0.2)
right = cv2.resize(right,None,fx=0.2,fy=0.2)

plot_images(left, right)

orb = cv2.ORB_create()

kp_left, des_left = orb.detectAndCompute(left, None)
kp_right, des_right = orb.detectAndCompute(right, None)

keypoints_drawn_left = cv2.drawKeypoints(left, kp_left, None, color=(0, 0, 255))
keypoints_drawn_right = cv2.drawKeypoints(right, kp_right, None, color=(0, 0, 255))

plot_images(left, keypoints_drawn_left, right, keypoints_drawn_right)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_left,des_right)

matches_drawn = cv2.drawMatches(left, kp_left, right, kp_right, matches, None, matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
plot_images(matches_drawn)

limit = 8
best = sorted(matches, key = lambda x:x.distance)[:limit]
# best_matches_drawn = cv2.drawMatches(left, kp_left, right, kp_right, best, None, matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# plot_images(best_matches_drawn)

# left_pts = np.array([kp_left[mat.queryIdx].pt for mat in matches])
# right_pts = np.array([kp_right[mat.trainIdx].pt for mat in matches])

# mask = LPM_filter(left_pts,right_pts)

# left_pts = left_pts[mask,:]
# right_pts = right_pts[mask,:]

# display2 = draw_match(left, right, left_pts, right_pts)
# cv2.imshow("after", display2)
# # press ESC to terminate imshow
# k = cv2.waitKey(0)
# if k == 27:
#     cv2.destroyAllWindows() 

left_pts = []
right_pts = []
for m in best:
    l = kp_left[m.queryIdx].pt
    r = kp_right[m.trainIdx].pt
    left_pts.append(l)
    right_pts.append(r)

M, _ = cv2.findHomography(np.float32(right_pts), np.float32(left_pts),cv2.RANSAC,10.0)

dim_x = left.shape[1] + right.shape[1]
dim_y = max(left.shape[0], right.shape[0])
dim = (dim_x, dim_y)

warped = cv2.warpPerspective(right, M, dim)
cv2.imshow("warp", warped)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows() 
# plot_images(warped)

comb = warped.copy()
# combine the two images
for k in range(left.shape[1]):
    for l in range(left.shape[0]):
        if left[l,k].all() != 0:
            comb[l,k] = left[l,k]
# crop
comb_gray = cv2.cvtColor(comb, cv2.COLOR_BGR2GRAY)
for j in range(dim_x):
    if comb_gray[:,j].sum() == 0:
        x_crop = j
        break
    else:
        x_crop = dim_x
for j in range(dim_y):
    if comb_gray[j,:].sum() == 0:
        y_crop = j
        break      
    else:
        y_crop = dim_y

comb = comb[:y_crop, :x_crop]
cv2.imshow("comb", comb)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows() 
# plot_images(comb)

