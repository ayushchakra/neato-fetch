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

An example of the software architecture's ability to detect matches is shown below:

![This is a demo of the FLANN matching algorithm in action.](images/flann_demo.gif)

## Fetch Workflow
The main workflow of the neato's fetch behavior is defined by the `NeatoState` finite state machine and executed by the `FetchNode` class. The workflow is as follows:
1. DRIVE_NEATO_START
    - The drive neato to start state essentially acts like a teleoperation algorithm for the user to use. It allows them to remotely drive the Neato to the desired start position. This state is ended by a KeyboardInterrupt triggered by the user.
2. ANALYZE_REF_IMAGE
    - This state is for establishing the ball and reference images to be tracked throughout the course of this behavior. This state is also dictated by a seperate finite state machine, called `DrawBoxState`. Once the state is initiated, the user is displayed a picture of the current frame, which becomes known as the reference frame. They are then prompted to select 4 points of interest. These 4 points become the bounds for the bounding box about the soccer ball and the person. Once all four have been selected, keypoints and image descriptors are extracted and saved from the respective bounding boxes. This state ends once all four corner positions have been inputted by the user and the image has been analyzed.
3. TRACK_BALL
    - This state manages the drive commands for following the soccer ball. The current frame is analyzed for all keypoints and image descriptors and then matched to the ball keypoints and image descriptors that were obtained in the previous state. If not enough matches are made, the Neato is instructed to continue moving in its current direction. If there are enough matches, the matching keypoints are averaged and the Neato is instructed to drive at a constant linear speed (0.2 m/s) and at an angular speed proportional to the difference between the average keypoint position and the center of the camera frame (is also scaled by a tunable constant). This state ends when the LiDAR detects that there is an object within a tunable distance of the Neato.
4. TRACK_PERSON
    - This state operates almost exactly the same as the previous state with two exceptions. When obtaining matches, the current frame is compared against the keypoints and image descriptors of the person, not the ball. Additionally, the state starts with a 180 degree rotation to initially guide the Neato back to the person. This state ends when the bump sensor is triggered.
5. CELEBRATION
    - Once the Neato successfully bumps into the person, it celebrates its accomplishment by doing a dance! This state ends once it completes the pre-computed dance.
6. DONE
    - Once the celebration has completed, the Neato is instructed to stop, signalling that the behavior has been completed.


## Design Decisions
moving in current direction when not enough matches
turn 180
lidar for ball, bump for person

## Results

## Reflection

## Next Steps