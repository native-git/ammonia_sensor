This program is designed to work with the Ammonia Sensing Platform. Much of the code is unique to that setup, but if you would like to see how it works, you can run it by providing the sample image as an argument. This program requires a linux system with some python dependencies. If you are experiencing any issues that say "ImportError: No module named <blank>", try googling the error, and installing the module using pip.

---------------------------

Stitched together by, Mike Sanzari

---------------------------

Explanation of the contents of this directory:

ammonia_camera_config.gpfl - configuration file for camera settings to provide consistent data for measurements
	* designed to work with "logitech Carl Zeiss Tesser HD 1080p usb-webcam"

ammonia_sensor.py - the actual python program that takes a picture, and calcuates concentration in ppm (among other things)

concentration_log.csv - Log file where ppm data, cage number, and time data are recorded.

/images/ - folder containing date-stamped images taken by this program

/csv_files/ - folder containing date-stamped RGB values measured from pictures taken by this program

sample_image.png - a sample of what the images taken with the whole system look like, so that you can try this program out yourself

README.txt - what you are currently reading :)

---------------------------

To use this program, do the following:

1. Move this folder "ammonia_sensor" (the one that this README.txt is in) to your home folder.

2. Give the ammonia_sensor.py program executable permissions using the following command:
	
	cd ~/ammonia_sensor && sudo chmod +x ammonia_sensor.py

3. Run the program, providing the path to the sample image as an argument using the following command:

	./ammonia_sensor.py -i sample_image.png

4. Select the cage number by pressing the "1","2", or "3" key on your keyboard.

5. Visually confirm the selected areas are correct (green boxes fall on solid regions of color, and not on things like the bottom of the can).

6. Press the "q" key twice to confirm the image, and calculate ppm (make sure you have clicked in the window where the image is present)
	*note, this won't work if you have clicked on the command line window (you should not see the letter q being typed)

7. The graph that pops up will display the Concentration vs. Hue curve as calculated by the sample areas, and the little traingle will represent the calculated ppm for your sample. Close this window by clicking on the red "x" in the upper left hand corner.

8. Before exiting, the program will print out the ppm value on the command line. This, the time-stamp, and the cage number will all be written to the concentration_log.csv file found in this folder as well. Finally, to see the raw RGB values for your regions of interest, see the .csv file with the proper time-stamp (most recent if you just ran it) in the "csv_files" folder in this directory.
