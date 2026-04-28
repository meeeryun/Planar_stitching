import cv2
import numpy as np
import os # To call image easily

# Call the image and Resize
RESIZE_WIDTH = 1500
image_path_template = './images3/image_{}.jpg'
images = []
for i in range(6):
    img = cv2.imread(image_path_template.format(i+1))
    if img is None:
        print(f"Error: image_{i+1}.png를 찾을 수 없습니다.")
        exit()
    h, w = img.shape[:2]
    ratio = RESIZE_WIDTH / float(w)
    target_size = (RESIZE_WIDTH, int(h * ratio))
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    images.append(img_resized)

# SIFT
sift = cv2.SIFT_create()

def get_homography(img1, img2):
    """두 이미지 사이의 특징점을 매칭하여 Homography 행렬을 계산하는 함수"""

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Find the Matching point
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: 
            good_matches.append(m)
            
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # RANSAC Homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    return None

# Grid Homography manual

H_list = [np.eye(3)] * 6

H_list[0] = get_homography(images[0], images[1]) # 0 -> 1
H_list[2] = get_homography(images[2], images[1]) # 2 -> 1
H_list[4] = get_homography(images[4], images[1]) # 4 -> 1

H_3to4 = get_homography(images[3], images[4])
H_list[3] = H_list[4] @ H_3to4 # 3 -> 4 -> 1

H_5to4 = get_homography(images[5], images[4])
H_list[5] = H_list[4] @ H_5to4 # 5 -> 4 -> 1

# Size of the result
h, w = images[1].shape[:2]
corners = []
for H in H_list:
    c = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    corners.append(cv2.perspectiveTransform(c, H))

all_corners = np.concatenate(corners, axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 10)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 10)

T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

pano_w, pano_h = x_max - x_min, y_max - y_min
pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

for i in range(6):
    final_H = T @ H_list[i]

    warped = cv2.warpPerspective(images[i], final_H, (pano_w, pano_h))
    
    mask = (warped > 0).astype(np.uint8) * 255
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    pano_mask = (pano > 0).astype(np.uint8) * 255
    pano_mask_gray = cv2.cvtColor(pano_mask, cv2.COLOR_BGR2GRAY)

    pano[mask_gray > 0] = warped[mask_gray > 0]

cv2.imwrite('./panorama_result.png', pano)
print("Panorama Complete: ./panorama_result.png")

# PTZ Viewer
view_w, view_h = 800, 600
x_pos, y_pos = pano.shape[1] // 2, pano.shape[0] // 2
zoom = 1.0
is_dragging = False
start_point, current_point = (0, 0), (0, 0)
dead_zone, ring_radius = 30, 80
BLUE_GLOW = (255, 200, 0)

def mouse_control(event, x, y, flags, param):
    global zoom, is_dragging, start_point, current_point
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0: zoom = min(5.0, zoom + 0.1)
        else: zoom = max(0.5, zoom - 0.1)
    elif event == cv2.EVENT_LBUTTONDOWN:
        is_dragging, start_point, current_point = True, (x, y), (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging: current_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False

cv2.namedWindow("PTZ Mouse Viewer")
cv2.setMouseCallback("PTZ Mouse Viewer", mouse_control)

while True:
    cur_w, cur_h = int(view_w / zoom), int(view_h / zoom)
    if is_dragging:
        dx, dy = current_point[0] - start_point[0], current_point[1] - start_point[1]
        if abs(dx) > dead_zone: x_pos += (dx / 15)
        if abs(dy) > dead_zone: y_pos += (dy / 15)
    
    half_w, half_h = cur_w // 2, cur_h // 2
  
    x_pos = max(half_w, min(x_pos, pano.shape[1] - half_w))
    y_pos = max(half_h, min(y_pos, pano.shape[0] - half_h))

    start_x, start_y = int(x_pos - half_w), int(y_pos - half_h)
    view = pano[start_y : start_y + cur_h, start_x : start_x + cur_w]
    if view.size == 0: continue
    view_res = cv2.resize(view, (view_w, view_h))

    if is_dragging:
        cv2.circle(view_res, start_point, ring_radius, BLUE_GLOW, 2)
        cv2.circle(view_res, start_point, dead_zone, BLUE_GLOW, -1)
        for d in [(0, -dead_zone), (0, dead_zone), (-dead_zone, 0), (dead_zone, 0)]:
            cv2.arrowedLine(view_res, start_point, (start_point[0]+d[0], start_point[1]+d[1]), (255,255,255), 2, tipLength=0.3)
        cv2.line(view_res, start_point, current_point, BLUE_GLOW, 2)
        
    cv2.putText(view_res, f"Zoom: {zoom:.1f}x", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE_GLOW, 2)
    cv2.imshow("PTZ Mouse Viewer", view_res)
    if cv2.waitKey(20) & 0xFF == 27: break

cv2.destroyAllWindows()