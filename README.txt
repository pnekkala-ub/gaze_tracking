1. Create a virtual environemnt with python 3.10
conda create -n <name> python=3.10 -y

Install dependencies like so
2. pip install --no-cache-dir -r requirements.txt
NOTE: You will have to move to the directory that contains the requirements.txt file. no-cache-dir ensures local cached files do not cause dependency conflicts.

To launch the program implementing algorithm1, execute the run.sh shell script like so
3. ./run.sh
NOTE: This will launch two windows. 1 displays the highlighted pupil and the other displays the binarized eyes.
      Upon the program receiving a keyboard interrupt (CTRL-C), you will observe a piechart of the user actions
      during the period the program was actively running. The program has to be terminated with a
      KeyboardInterrupt to be able to see the insights. Wait atleast a minute for the program to collect enough data.


sample_output_1 and sample_output_2 in the parent directory contains example outputs of the application.

To test the program implementing algorithm2, execute the run2.sh shell script like so
4. ./run2.sh
NOTE: This program launches the video capture omn your PC and shows the functioning of 
      pupil tracking using the YOLOv8 trained model for inference. It will not offer insights.
      That component is not implemented considering it performs poorly at the task of identifying 
      the puils. It is included to demonstrate the experimental results and discuss the observations
      and drawbacks of this algorithm.

Rest of the files are helper modules coded to prepare data, annotate data and train the model.
They are included to demostrate the coding effort that was put into the project.


mail id: pnekkala@buffalo.edu