We can use this program to detect lane on road in any videos. As you can see, it can draw the ROI to let you know the direction you need to drive. Though, it is still defects in this code. But it still can guide you how to know the region.
Still need some improvement to optimize this code. When you use this code, it will give you an output video when you give it a video you want to detect.
------------------------Code explanation-----------------------
This code I let it read the video you give, and it will process every frame in this video to detect the edge, covert it to bird-view, slide the windows to find lanes on road, and finally we put and pile them to the origianl.
threshold_pipeline for detecting the region between the lanes, and I use the sobel edge detection and HLS color space to do the sobel X when frame is gray-level.
wrap_perspective to make the tilted-view road to convert it into bird-view to easily detect the lanes.
find_lane_lines for using slide windows to find the left and the right in the binary bird-view.
draw_lane for coloring the ROI and to pile them on the original image and let it green.
process_video to process every frame to make them a video.
