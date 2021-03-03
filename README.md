# Color Segmentation using Gaussian Mixture Models 

## Introduction 


<img src="outputs/buoy.gif" width="640" height="480">

This project performs colour segmentation underwater with the use of methods like implementation of Gaussian Mixture Models and Expectation Maximization.
The video provided includes 3 colours of Yellow, Orange and Green. However, due the variation of the shades of colour in each frame, conventional techniques could not be implemented for segmentation. Thus, we use the Gaussian techniques to train the model using the data set that is generated to help the model detect the color based on the variations. 

The output images of all three colours being detected are shown along with the histograms for each colour ``outputs/Buoy_detection1.png``

The final video with the overlap of the detection frames for each colour is in ``outputs/3D_gauss.avi``

The report containing the details of the outputs and the plots is [here](https://github.com/kmushty/Buoy_Detection/blob/main/Project_3_Report.pdf) 


## Dependencies

The following are the project dependencies:
- OpenCV 3.4.2 or above
- Python 3.5

## Code Execution

In order to implement the code:
- Clone the repo
- Run using the following command in the command line ``python3 GMM.py``
- The code will begin to run and display a set of instructions. The user must click on the image display according to the requested color in the title of the window.
- Left click on the buoy color that is requested in the title of the window. If the buoy is partially off the screen or completely out of the image, Right click anywhere on the display. Hit Space Bar after making a selection to proceed. The selected coordinate will appear in the command window.
- The code also prints out the graphs for the histograms for each colour and the 1-D gaussian outputs.
- To generate the individual colour outputs, run the Seg_green, Seg_red and Seg_yellow respectively.

## Output 

<img src="outputs/Buoy_detection1.png" width="640" height="480">
