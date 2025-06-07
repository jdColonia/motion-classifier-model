# Data Collection for Motion Classifier

This directory contains the collected and processed data for training the motion classification model.

## Data Collection Process

The data was collected through video recordings of the following movements:

- Walking forward
- Walking backward
- Sitting down
- Standing up
- Turning

### Recording Methodology

1. A mobile phone camera was used to record the movements
2. Each movement was recorded multiple times by multiple people to obtain data variability
3. Recordings were made in a controlled environment with good lighting
4. An adequate distance was maintained from the camera to properly capture the movement

## Data Structure

### `/processed` Folder

Contains the processed data organized by movement type:

- `/processed/walk_forward/` - Walking forward videos
- `/processed/walk_backward/` - Walking backward videos
- `/processed/sit_down/` - Sitting down videos
- `/processed/stand_up/` - Standing up videos
- `/processed/turning/` - Turning videos

It also contains the combined dataset:

- `combined_dataset.csv` - Final dataset containing all processed movements with their landmarks and labels

## Data Processing

The data was processed using MediaPipe Pose to extract body landmarks. The process included:

1. Recording videos for each movement
2. Processing videos using MediaPipe for pose detection
3. Extracting relevant landmarks
4. Combining all data into a single CSV dataset

### Important Landmarks

The following key points were captured:

- Nose (0): For orientation and direction
- Shoulders (11-12): Upper body posture
- Hips (23-24): Base body position
- Knees (25-26): Leg flexion
- Ankles (27-28): Feet position
- Feet (31-32): Lower extremity movement

### Combined Dataset Structure

The `combined_dataset.csv` file contains:

- Coordinates (x, y, z) and visibility of each landmark
- Numerical label for the movement
- Movement name
- Only includes frames where pose was correctly detected (pose_detected = True)
