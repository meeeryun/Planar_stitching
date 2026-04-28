# Planar_stitching
My planar_stitching

# Core Code Analysis
### 1. Image Preprocessing & Resizing
Since high-resolution images require significant computational power, resizing is a crucial step for efficient processing.
```ratio = RESIZE_WIDTH / float(w)
target_size = (RESIZE_WIDTH, int(h * ratio))
img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
```
- Resizes all input images to a fixed width (RESIZE_WIDTH) while maintaining the original aspect ratio.
- Purpose:
  (1) Optimization: Speeds up SIFT feature extraction and matching.
  (2) Memory Management: Prevents Out-Of-Memory (OOM) errors when creating the final large panorama.
  (3) Quality: Uses ```INTER_AREA``` interpolation to minimize aliasing and preserve feature details during downscaling.

### 2. Feature Detection & Matching
We utilize the SIFT algorithm, as discussed in class, to find correspondences between overlapping images.
```
sift = cv2.SIFT_create() # Initialize SIFT
kp, des = sift.detectAndCompute(img, None) # Detect Keypoints and compute Descriptors
```
- SIFT: Extracts robust features that are invariant to scale, rotation, and illumination changes.
```
matches = bf.knnMatch(des1, des2, k=2) # k-Nearest Neighbor Matching
for m, n in matches:
    if m.distance < 0.75 * n.distance: # Lowe's ratio test
        good_matches.append(m)
```
- Lowe's Ratio Test: Compares the distance of the best match to the second-best match. This effectively filters out ambiguous matches (outliers), ensuring only reliable correspondences are used.

### 3. Homography & RANSAC (Geometric Transformation)
Instead of using high-level APIs like cv2.Stitcher, we manually compute the planar projection matrix.
```
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```
- Homography (H): A 3x3 matrix that maps points from one plane to another.
- RANSAC: A robust estimation method that iteratively selects random samples to find the best model. It ensures an accurate transformation even if some initial matches are incorrect.

### 4. 3x2 Grid Stitching StrategyTo align 6 images in a $3 \times 2$ grid, we set Image 1 as the "Anchor" (global coordinate origin).
- Logic:
  (1) Images 0, 2, and 4 are directly aligned to Image 1.
  (2) Chain Transformation: ```H_list[3] = H_4to1 @ H_3to4```. Image 3 is first mapped to Image 4, then transformed to Image 1's coordinate system using matrix multiplication. This ensures       all 6 images reside in a single unified plane.

### 5. Warping & Canvas Integration
This step transforms all images into the final panorama canvas without cropping issues.
```
T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) # Translation Matrix
warped = cv2.warpPerspective(images[i], T @ H_list[i], (pano_w, pano_h))
```
- Translation (T): Prevents images from being cut off by shifting negative coordinates (caused by homography) into the positive canvas area.
- warpPerspective: Applies the final accumulated transformation matrix to project each image onto the panorama.

### 6. Interactive PTZ (Pan-Tilt-Zoom) Viewer (Only-using MOUSE)
A custom feature that allows users to explore the high-resolution panorama interactively.
```
view = pano[start_y : start_y + cur_h, start_x : start_x + cur_w] # Slicing/Cropping
view_res = cv2.resize(view, (view_w, view_h)) # Resize to fit Window
```
- Implementation: Dynamically slices a sub-region of the panorama based on the mouse position (```x_pos```, ```y_pos```) and the ```zoom``` level.
- User Experience: Features a "Dead Zone" to prevent jittery movements and a "Blue Glow" UI for intuitive control.

# 후기:

제가 이번 프로젝트를 진행하면서 생각이 들었던 것은 몇 손가락 안에 들 정도로 가장 힘든 순간으로 손 꼽을 것 같습니다. 이유를 하나씩 말해보자면,
1. 기본적으로 고해상도 사진을 사용합니다. 그렇기 때문에 랩탑의 성능이 정말 중요한데 저의 랩탑 성능은 그리 좋지 않아서 더더욱 힘들었습니다.
2. 처음에는 원래 Planar가 아닌 spherical view를 적용해서 가로로 파노라마를 만드려고 했습니다. 그리고 PTZ Viewer를 군대에서 사용한 적이 있어서 유용한 뷰어라고 생각해 PTZ까지 구헌을 하려고 했던겁니다. 그렇지만 1번 이유도 있었고 제가 7세트가 넘는 사진들을 찍어서 거의 70장 가까이 찍었는데도 제가 원하는만큼의 고퀄리티로 만들어지지 않았습니다. 항상 이미지들이 아예 검게 타서 날라가버리거나, 공간이 뒤틀린 것 같이 구도가 깨져서 나왔기 때문입니다. 개인적으로 생각을 했던 것은, 직접 구현을 통해서는 한계가 있다고 생각합니다. 특히 Spherical view나 Cylindrical View는 차원이 더 높은 관점에서 프로그램이 실행되어야 하기 때문에 더 조정되고 세밀한 상위 API를 사용하는 것이 올바르다고 생각했습니다.
3. 직접 Homography, wrapPerspective 등을 사용한 직접 구현과 Stitcher를 둘 다 해본 결과, Stitcher가 정말 좋다는 걸 느꼈습니다. Stitcher API를 사용했을 때는 처음에 사용했던 사진 세트들도 너무 완벽하게 구현이 되었었습니다. PTZ까지 너무 자연스럽게 적용되어서 끊김 하나없이 Zoom과 카메라 이동까지 가능한 수준이었습니다. 하지만 직접 구현을 사용해보니 위에서 언급했듯이 이미지들이 검게 타버리거나 점들을 Matching 하는 과정들이 고성능을 필요로 하고 아예 못 찾을 때도 있어서 구도가 깨져버립니다.

코딩을 하면서 처음으로 함수를 대신해서 직접 구현을 해본 거 같은데 그 중에서도 이번 프로젝트는 많이 힘들었습니다. 이런 함수들이 얼마나 세밀한 작업들을, 얼마나 최적화 해서 내부에서 수행중인건지 이번 기회로 뼈저리 느꼈습니다.
