import cv2

#exercise 1
image1= cv2.imread('image1.png')
cv2.imshow('Image1 - Original', image1)
cv2.waitKey(0)

image1_gray= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image1 - Grayscale', image1_gray)
cv2.waitKey(0)

image1_rotated= cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite('image1.rotated.png', image1_rotated)

image1_resized= cv2.resize(image1, (300, 300))
cv2.imshow('Image1 - Resized', image1_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#exercise 2

image2= cv2.imread('image2.jpg')
gaussian= cv2.GaussianBlur(image2, (5,5),0)
cv2.imshow('gaussian',gaussian)
cv2.waitKey(0)

median=cv2.medianBlur(image2,5)
cv2.imshow('median',median)
cv2.waitKey(0)

laplacian= cv2.Laplacian(image2,cv2.CV_64F)
cv2.imshow('laplacian',laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

#exercise 3 

image3= cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)
sobel_x= cv2.Sobel(image3,cv2.CV_64F,1,0,ksize=3)
sobel_y= cv2.Sobel(image3,cv2.CV_64F,0,1,ksize=3)
sobel_added= cv2.magnitude(sobel_x,sobel_y)
cv2.imwrite('image3.sobel.jpg',sobel_added)

canny_low = cv2.Canny(image3, 50, 150)
cv2.imwrite('image3.canny_low.jpg', canny_low)
canny_high = cv2.Canny(image3, 100, 200)
cv2.imwrite('image3.canny_high.jpg', canny_high)

laplacian_edges = cv2.Laplacian(image3, cv2.CV_64F)
cv2.imwrite('image3.laplacian.jpg', laplacian_edges)

#exercise 4

image4 = cv2.imread('image4.jpg', cv2.IMREAD_GRAYSCALE)

_, otsu_thresh = cv2.threshold(image4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('image4.otsu.jpg', otsu_thresh)

adaptive_mean = cv2.adaptiveThreshold(image4, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
adaptive_gaussian = cv2.adaptiveThreshold(image4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('image4.adaptive_mean.jpg', adaptive_mean)
cv2.imwrite('image4.adaptive_gaussian.jpg', adaptive_gaussian)