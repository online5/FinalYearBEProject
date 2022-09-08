# Final Year BE Project

Implemented an Automatic number plate recognition system(ANPR) to authorize vehicles in a residential society.
![block diagram](https://user-images.githubusercontent.com/40465321/122954395-06107480-d39d-11eb-9b05-78bc4371e3ec.png)

**Project Breakdown**:

Module A- Image Processing:

    A.1 Movement Detection-Background Substraction
    A.2 Vehicle/Non Vehicle Classification-YOLOv3
    A.3 Plate Localization-Aspect Ratio Based Localization
    A.4 Character Recognition-Skew correction, Character segmentation, NN based classification

Module B- WebApp Development:

    B.1 Frontend Development- HTML, CSS, Bootstrap
    B.2 Backend Developement- Python, Django, SQLite database

**Libaries**:

    Python 3.8.5
    Django==3.1.4
    numpy==1.18.5
    opencv-python==4.4.0.46
    pytesseract==0.3.7
    scikit-image==0.17.2
    Keras==2.4.3
    tensorflow==2.3.1

**Project Description**:

For description and working demo of the project, refer this link
https://drive.google.com/file/d/1QxaeCh9ShxnVnVMUmI1JL9qZHRZSW-hh/view?usp=sharing
(The above video is divided into two part 1: Description 2: Working demo, please wait while the video is loading also note the demo starts at 17.0 min).

**Installation Steps:**


1: Clone the repository.

2: To run this application download the required machine learning models from this link: https://drive.google.com/drive/folders/1DZGw1bR48PlTRuQR_7qhAgNK3qpvRsbX?usp=sharing and paste into this directory FinalYearBEProject/django-project/ANPR_System/web/ML_Models (create ML_Models empty folder if not created earlier).

3: Make sure to install above libraries.

4: Open terminal at **FinalYearBEProject/django-project/ANPR_System/manage.py** and run the project using **python manage.py runserver** command.

5: Open url-http://127.0.0.1:8000/ (username=fantasticfour password=binod_op).
(Note- The operations of the project can be managed using django admistration)

6: To access and use email functionality in the application, go to the path **FinalYearBEProject/django-project/ANPR_System/mysite/settings.py**, open the settings.py file, and set **EMAIL_HOST_USER** and **EMAIL_HOST_PASSWORD** variables in compliance with your mailing account credentials. 
(Make sure to allow third-party apps access while running this application.



## Contributors: 
   
   [Tanmay Shinde](https://github.com/tanmayshinde007).
   [Atharva Jadhav](https://github.com/atharva21jadhav). [Omkar Dongre](https://github.com/omkardongre). 
   
   


