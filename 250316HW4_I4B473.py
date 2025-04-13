import cv2
import numpy as np

# ---------- STEP 1: Threshold pipeline ----------
def threshold_pipeline(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel x on gray
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    scaled_sobel = np.uint8(255 * np.absolute(sobelx) / np.max(np.absolute(sobelx)))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    # S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 80) & (s_channel <= 255)] = 1

    # L channel Sobel
    sobel_l = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    scaled_l = np.uint8(255 * np.absolute(sobel_l) / np.max(np.absolute(sobel_l)))
    sobel_l_binary = np.zeros_like(scaled_l)
    sobel_l_binary[(scaled_l >= 30) & (scaled_l <= 100)] = 1

    combined = np.zeros_like(sobel_binary)
    combined[(sobel_binary == 1) | (s_binary == 1) | (sobel_l_binary == 1)] = 1
    return combined

# ---------- STEP 2: Perspective transform ----------
def warp_perspective(img):
    h, w = img.shape[:2]
    src = np.float32([
        [200, h], [1080, h], [700, 450], [580, 450]
    ])
    dst = np.float32([
        [300, h], [980, h], [980, 0], [300, 0]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, M, Minv

# ---------- STEP 3: Sliding window ----------
def find_lane_lines(binary_warped, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    out_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8)
    out_img[binary_warped == 1] = [255, 255, 255]

    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else [0, 0, 0]
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else [0, 0, 0]

    return left_fit, right_fit

# ---------- STEP 4: Draw lane ----------
def draw_lane(original_img, binary_warped, left_fit, right_fit, Minv):
    h, w = binary_warped.shape
    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    color_warp = np.zeros((h, w, 3), dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, [np.int32(pts)], (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result

# ---------- STEP 5: Process video ----------
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        warped, M, Minv = warp_perspective(frame)
        binary = threshold_pipeline(warped)
        left_fit, right_fit = find_lane_lines(binary)
        result = draw_lane(frame, binary, left_fit, right_fit, Minv)
        out.write(result)

    cap.release()
    out.release()

process_video("LaneVideo.mp4", "lane_detected_output.mp4")