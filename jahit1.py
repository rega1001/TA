import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.neighbors import KDTree
import random

# RANSAC Estimator
def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

# Matcher LPM
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

    return idx #mask.flatten().astype('bool')

def matcher(kp1, des1, kp2, des2):
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)

    # Apply ratio test
    limit = 8
    matches = sorted(matches, key = lambda x:x.distance)[:limit]

    # left_pts = np.array([kp1[mat.queryIdx].pt for mat in matches])
    # right_pts = np.array([kp2[mat.trainIdx].pt for mat in matches])
    # idx = LPM_filter(left_pts,right_pts)

    matche = []
    for mat in matches:
        matche.append(list(kp1[mat.queryIdx].pt + kp2[mat.trainIdx].pt))
    matches = np.array(matche)
    return matches

def plot_images(*imgs, figsize=(30,20), hide_ticks=False):
    f = plt.figure(figsize=figsize)
    width = np.ceil(np.sqrt(len(imgs)))
    height = np.ceil(len(imgs) / width)
    for i, img in enumerate(imgs, 1):
        ax = f.add_subplot(int(height), int(width), i)
        if hide_ticks:
            ax.axis('off')
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

image_paths=glob.glob('dummy\*.png')
imgs = []
n = len(image_paths)
for i in range(n):
	imgs.append(cv2.imread(image_paths[i]))
	imgs[i]=cv2.resize(imgs[i],None,fx=0.5,fy=0.5)
pref = imgs[0]

# Deklarasi metode
orb = cv2.ORB_create()

# Mulai loop
for i in range(1,n):
    left = pref
    right = imgs[i]

    kp_left, des_left = orb.detectAndCompute(left, None)
    kp_right, des_right = orb.detectAndCompute(right, None)

    # keypoints_drawn_left = cv2.drawKeypoints(left, kp_left, None, color=(0, 0, 255))
    # keypoints_drawn_right = cv2.drawKeypoints(right, kp_right, None, color=(0, 0, 255))
    # plot_images(left, keypoints_drawn_left, right, keypoints_drawn_right)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_left,des_right)

    # matches_drawn = cv2.drawMatches(left, kp_left, right, kp_right, matches, None, matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # plot_images(matches_drawn)

    limit = 4
    matches = sorted(matches, key = lambda x:x.distance)[:limit]

    # best_matches_drawn = cv2.drawMatches(left, kp_left, right, kp_right, best, None, matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # plot_images(best_matches_drawn)

    left_pts = np.array([kp_left[mat.queryIdx].pt for mat in matches])
    right_pts = np.array([kp_right[mat.trainIdx].pt for mat in matches])
    # idx = LPM_filter(left_pts,right_pts)
    # left_pts = left_pts[mask,:]
    # right_pts = right_pts[mask,:]
   
    # matches = matcher(kp_left, des_left, kp_right, des_right)
    # inliers, M = ransac(matches, 0.5, 2000)
    M, _ = cv2.findHomography(np.float32(right_pts), np.float32(left_pts))

    dim_x = left.shape[1] + right.shape[1]
    dim_y = left.shape[0] + right.shape[0]
    dim = (dim_x, dim_y)

    warped = cv2.warpPerspective(right, M, dim)

    # plot_images(warped)

    comb = warped.copy()
    # combine the two images
    for j in range(left.shape[1]):
        for k in range(left.shape[0]):
            if left[k,j].all() != 0:
                comb[k,j] = left[k,j]
    # crop
    comb_gray = cv2.cvtColor(comb, cv2.COLOR_BGR2GRAY)
    for j in range(left.shape[1],dim_x):
        if comb_gray[:,j].sum() == 0:
            x_crop = j
            break
        else:
            x_crop = dim_x
    for j in range(left.shape[0],dim_y):
        if comb_gray[j,:].sum() == 0:
            y_crop = j
            break      
        else:
            y_crop = dim_y

    pref = comb[:y_crop, :x_crop]


cv2.imshow('Hasil', pref)
cv2.waitKey(0)
cv2.destroyAllWindows()