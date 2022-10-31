# Neato Fetch
## Project Overview
The purpose of this project was to develop a software architecture that relies on visual elements from a robot's surrounding to influence its motion. More specifically, we chose to train a Neato to retrieve (or fetch) a ball in motion and then return to the person who initiated that motion and then "celebrate" once it successfully completed the task.

In order to accomplish this task, we needed to be able to accurately track the position of the ball within the frame of the Neato's camera. Then, once the ball has been reached, we needed a way to locate and drive to the person who initiated the motion, also based on visual features. This was accomplished by taking a reference image and analyzing it for key points relevant to the ball and the person, which would then be monitored at a frame-by-frame level. The full impementation, which is described below, can be found in this [github repository](https://github.com/ayushchakra/neato-fetch).

## Problem-Solving Approach
In order to successfully develop the neato's fetch behavior, we chose to implement an iterative approach in terms of algorithmic robustness and complexity, starting from a simple, inconsistent color tracking algorithm, and scaling to developing an image descriptor tracking algorithm that is far more complex and consistent. This was done in three main stages: color masking, contour detection, and image descriptor tracking. 

### Color Masking

### Contour Detection

### Image Descriptor Tracking
The final attempted implementation of object tracking was to use a pre-built image descriptor detection and matching algorithm. The selected image descriptor algortihm was Oriented FAST and Rotated BRIEF (ORB). Essentially, ORB is a fusion of Features from Accelerated Segment Test (FAST) keypoint detector and Binary Robust Independent Elementary Features (BRIEF) descriptor. The benefit of ORB is that it is less computationally costly as compared to other common algorithms, such as SIFT and SURF, but still maintains a comparable level of accuracy. This was important as we wished to do object tracking in real-time, which requires us to minimize the lag time between analysis of the current frame and sending commands to the Neato. The algorithm detects keypoints and descriptors based on the relative brightness shift at a given pixel relative to the surrounding pixels.

The ORB algorithm was applied to the initial frame to determine the keypoints and descriptors of interest. At first, this was done by analyzing the entire frame, but hindered our ability to track two different, distinct features: a person and a ball. Thus, we implemented a bounding box feature, where the user defines two bounding boxes that define where the person is located and where the ball is located. Only keypoints and descriptors founding within these respective boundaries were stored. Then, each subsequent frame is analyzed using ORB and compared to the corresponding reference keypoints and descripors based on which state the Neato is in (compared to ball features if following the ball and compared to the person features if returning to the person).

In order to find matches, we implemented the Fast Library for Approximate Nearest Neighbors. Once again, speed was essential to our software's success, which is why FLANN was chosen as opposed to a more traditional algortim, like a Brute-Force Matcher. The algorithm is optimized for fast nearest neighbor searches, especially on large data sets, reducing runtime. Using FLANN, we found matches between the current frame and the relevant comparator, which instructed the drive commands that were sent to the Neato.

## Fetch Workflow

## Design Decisions

## Results

## Reflection

## Next Steps