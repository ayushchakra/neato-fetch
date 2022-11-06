# Neato Fetch
## Project Overview
The purpose of this project was to develop a software architecture that relies on visual elements from a robot's surrounding to influence its motion. More specifically, we chose to train a Neato to retrieve (or fetch) a ball in motion and then return to the person who initiated that motion and then "celebrate" once it successfully completed the task.

In order to accomplish this task, we needed to be able to accurately track the position of the ball within the frame of the Neato's camera. Then, once the ball has been reached, we needed a way to locate and drive to the person who initiated the motion, also based on visual features. This was accomplished by taking a reference image and analyzing it for key points relevant to the ball and the person, which would then be monitored at a frame-by-frame level. The full impementation, which is described below, can be found in this [github repository](https://github.com/ayushchakra/neato-fetch).

## Problem-Solving Approach
In order to successfully develop the Neato's fetch behavior, we chose to implement an iterative approach in terms of algorithmic robustness and complexity, starting from a simple, inconsistent color tracking algorithm, and scaling to developing an image descriptor tracking algorithm that is far more complex and consistent. This was done in three main stages: color masking, contour detection, and image descriptor tracking. 

### Color Masking
Our initial method for isolating and tracking the ball was using color masking to create a binary image. This would have been done using OpenCV's inRange function, which creates a mask of an image based on a specific RGB color range. By doing this, a binary image of the specified color range can be created. 

If we use a distinctly colored ball (i.e. orange) in a plain environment, it is simple to track the ball. With a binary image, we only need to find the average x-coordinate of all the white pixels in a given frame. Subtracting this value from the center x-coordinate will give us the distance the Neato needs to move. Then, we can divide that number by a proportinal constant to determine the speed at which the Neato needs to turn. We can ignore the y-coordinate because we are only moving in regards to one axis.

This tracking method for the ball is what was implemented in the Neato Soccer exercise, and we decided we wanted to explore a different avenue of computer vision.

### Contour Detection

The second ball tracking method we implmented revolved around contour matching. This was accomplished using OpenCV's findContours, approxPolyDP, and drawContour functions. 

Using the color masking method described in the previous section, a binary image analyzed using the findCountour function. This function identifies contours in the image, and in conjunction with the approxPolyDP function, can identify the specific shape the contours create with reasonable accuracy. However, this method was not accurate in the dynamic environment of the moving ball and an inconsistant background. This gave the contours a significant amount of background noise, which lead us to believe that this method was not the optimal way of tracking the ball.

Erosion and dilation techniques were also applied, but still failed at reasonably identifying the countours and shapes in the image.


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
There were three interesting design decisions that we made during this project.

Two design decisions we made prior to physically testing the Neato tracking were:
1. Continuing to turn the Neato in current direction when not enough matches were available.
    - If the object that the Neato is tracking is out of frame or there are too little matches, the Neato will continue to move at the last speed that was published to it. Since the object is only moving faster than the Neato is turning, but is still in the same direction, the Neato will eventually turn enough to get the object in frame and thus obtain image descriptors.
2. After completing the `TRACK_BALL` state, the Neato will turn 180 degrees.
    - This makes it easier for the Neato to create find keypoints on the person. With a direct 180 degree turn, the Neato's camera is facing the person that was identified in the `DRIVE_NEATO_START` state. This mitigates the need of a FIND_PERSON state.

While testing with the Neato, we found that one of our previous assumptions conecrning the Neato's bump sensor was incorrect. We believed that the Neato's bump sensor would have been able to register when the Neato bumped into the ball. However, in practice, the Neato was travelling too slow while the ball was moving in the opposite direction to activate the bump sensor. To identify when the Neato was "kicking" the ball, we used the Neato's LiDAR to simulate the bump. By detecting when the Neato was within a certain range of the ball, we are able to kick the ball and change the state. We continued to use the bump sensor to identify when the person was reached.

## Results

A demo video can be found [here](https://www.youtube.com/shorts/1TKgzT15Tm8).

## Reflection

A challenge we faced during this project was the implementation of ORB and FLANN. While the OpenCV documentation could have been more forgiving, we worked through it together to understand which functions return keypoints, descriptors, etc.

To improve this project, we should optimize our code so we can be able to increase the speed of the Neato. If our code was less computationally intensive, we would be able to sucessfully increase the speed of the Neato.

Also, adding visualizations to the video feeds from the Neato would be helpful in the future. By drawing bounding boxes around our ROIs, we would be able to better see what the Neato is tracking and what areas we define to track.


## Future

Learning, understanding, and applying object descriptors (specifically ORB) was very applicable to computer vision research and projects that we are currently working on. In the RoboLab, Evan can implement the use of ORB on Hummingbird and in a project outside of Olin. Ayush is able to use these skills for a project he is currently working on for drone tracking.
