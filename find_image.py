'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('lion2.png')

# Convert to greyscale
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(grey, (5, 5), 0)

# Apply Sobel operator to detect gradients
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Compute gradient magnitude and scale
grad_magnitude = cv2.magnitude(grad_x, grad_y)
grad_magnitude = np.uint8(255 * grad_magnitude / np.max(grad_magnitude))

# Apply threshold to get binary image
_, binary = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract key points (approximated contours)
key_points = []
for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjust epsilon for precision
    approx = cv2.approxPolyDP(contour, epsilon, True)
    key_points.append(approx)

# Draw key points on the image
image_with_keypoints = image.copy()
for points in key_points:
    for point in points:
        cv2.circle(image_with_keypoints, tuple(point[0]), 3, (0, 0, 255), -1)

# Save key points to a file (optional)
with open('key_points.txt', 'w') as file:
    for points in key_points:
        for point in points:
            file.write(f"{point[0][0]},{point[0][1]}\n")

# Show the results
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title('Greyscale Image')
plt.imshow(grey, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Gradient Magnitude')
plt.imshow(grad_magnitude, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Key Points Detected')
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))

plt.show()


'''
'''
# attempt 2


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('lion2.png')

# Convert to greyscale
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(grey, (5, 5), 0)

# Apply Sobel operator to detect gradients
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Compute gradient magnitude and scale
grad_magnitude = cv2.magnitude(grad_x, grad_y)
grad_magnitude = np.uint8(255 * grad_magnitude / np.max(grad_magnitude))

# Apply threshold to get binary image
_, binary = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract key points (approximated contours)
key_points = []
for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjust epsilon for precision
    approx = cv2.approxPolyDP(contour, epsilon, True)
    key_points.extend(approx)

# Detect corners using Shi-Tomasi method
corners = cv2.goodFeaturesToTrack(grey, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)

# Add the detected corners to key points
for corner in corners:
    x, y = corner.ravel()
    key_points.append([[x, y]])

# Draw key points on the image
image_with_keypoints = image.copy()
for point in key_points:
    cv2.circle(image_with_keypoints, tuple(point[0]), 3, (0, 0, 255), -1)

# Save key points to a file (optional)
with open('key_points.txt', 'w') as file:
    for point in key_points:
        file.write(f"{point[0][0]},{point[0][1]}\n")

# Show the results
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title('Greyscale Image')
plt.imshow(grey, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Gradient Magnitude')
plt.imshow(grad_magnitude, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Key Points Detected')
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))

plt.show()
'''


# attempt 3
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


what_to_load = 'lion.png'

# Load the image
image = cv2.imread(   what_to_load   )

# Convert to greyscale
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(grey, (5, 5), 0)

# Apply Sobel operator to detect gradients
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Compute gradient magnitude and scale
grad_magnitude = cv2.magnitude(grad_x, grad_y)
grad_magnitude = np.uint8(255 * grad_magnitude / np.max(grad_magnitude))

# Apply threshold to get binary image
_, binary = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)

# Get coordinates of white pixels (high gradient areas)
coordinates = np.column_stack(np.where(binary > 0))

x_coords , y_coords = np .where( binary > 0 )

# Fit linear regression to the coordinates
X = coordinates[:, 1].reshape(-1, 1)  # x-coordinates
y = coordinates[:, 0]  # y-coordinates


# Initialize and fit the model
model = LinearRegression().fit(X, y)

# Predict y-values for a range of x-values to create a line
x_values = np.arange(binary.shape[1]).reshape(-1, 1)
y_values = model.predict(x_values)

# Create an image to display the linear fit
image_with_fit = image.copy()
for x, y in zip(x_values, y_values):
    cv2.circle(image_with_fit, (int(x), int(y)), 1, (0, 0, 255), -1)


# introducing a laplacian filter to smooth out the edges
laplacian = cv2 . Laplacian( grad_magnitude , cv2.CV_64F )
laplacian = np . uint8( np . abs( laplacian ) )

# extracting the points with bitmaping that is essentially zeroing and onening 
# keep all > 0
threshold = 30
_, binary_laplacian = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)

# Extract coordinates of significant points (white points in binary_laplacian)
points = np.column_stack(np.where(binary_laplacian > 0))





# Show the results
plt.figure(figsize=(10, 10))
'''
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title('Greyscale Image')
plt.imshow(grey, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Gradient Magnitude')
plt.imshow(grad_magnitude, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Linear Fit on High Gradient Areas')
plt.imshow(cv2.cvtColor(image_with_fit, cv2.COLOR_BGR2RGB))

plt.show()
'''


rotated_x_coords = y_coords
rotated_y_coords = -x_coords + max(x_coords)

plt.figure(figsize=(10, 10))
plt.scatter(   rotated_x_coords   ,    rotated_y_coords   ,    c='r'  ,    s=0.7  )  # c='r' sets color to red, s=1 sets size of points
#plt.gca().invert_yaxis()  # Invert y-axis to match the image coordinates
plt.title('Detected Key Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


plt .figure( figsize=( 10 , 10 ) )
plt . title( "Grad magn. with laplacian filter" )
plt . imshow( grad_magnitude , cmap="Blues" )
plt . show()

plt . figure( figsize=( 12 , 8 ) )
plt . title( "points from laplacian filter over grad magnitude" )
plt . imshow( binary_laplacian , cmap = 'gray' )
plt . scatter( points[:,1] , points[:,0] , c = 'r' , s = 1 )
plt . show()


rotated_coords = list(zip(rotated_x_coords, rotated_y_coords))

# Define the output file path
output_file_path = 'rotated_coordinates.txt'

if what_to_load == 'wolf.png':
    with open( 'wolf_points' , 'w' ) as file:
        for cX , cY in rotated_coords:
            file . write( f'{cX},{cY}\n'  )
else:
    # Write the rotated coordinates to the file
    with open(output_file_path, 'w') as file:
        for x, y in rotated_coords:
            file.write(f"{x},{y}\n")