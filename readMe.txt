-------------------------------------------------------

Source Code
-------------------------------------------------------


Specification of dependencies：

The code is written in Python 3.8. Based on the specified software version, OpenCV 3.0, scipy 1.5, Keras 2.0, Tensorflow 2.4, scikit-image 0.17, scikit-learn 0.23 libraries are required for the environment setting. All files need to be on the same directory in order for the algorithms to work.

To run the algorithm, change the path to the current directory in the command window, and run the [main.py] file:

main.py
The main method that implements the proposed algorithm on a task sequence to perform semantic segmentation and refining segmentations based on user interactions. Results will be generated on console, with the visualized ground truth, segmentation results and saved to output folder.

The main methods call the following functions:

1. fmodel.py
Includes methods that define the architecture of the network, customized block and layers of the network.

2. finteract.py
Includes functions for collecting user annotations from user interface.

3. fmisc.py
Includes functions for importing and exporting model parameters.

4. fprocess.py
Includes functions initializing model parameters, and reading, writing and processing images.

5. fsoften.py
Includes utility methods for label softening.