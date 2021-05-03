#NOTES: To debug uncomment cv2.imwrite function call statements in complete_anpr function.
#Web Module Imports
from mysite.settings import EMAIL_HOST_USER
from datetime import datetime
from django.views import generic
from django.utils import timezone
from django.shortcuts import render
from django.urls import reverse_lazy
from django.shortcuts import redirect
from django.core.mail import EmailMessage
from .models import User, Visitor, Resident, Videos
from django.http import HttpResponse, HttpResponseRedirect
from django.core.mail import BadHeaderError, get_connection
from .forms import ResidentForm, RemoveForm, UpdateForm, VideoForm, EmailForm, VisitorForm

#Image processing module imports
import os
import cv2
import base64
import numpy as np
from web.End2EndANPR import *
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

#Global Variables
classes=[]
first_letter_model=None
first_letter_labels=None
second_letter_model=None
second_letter_labels=None
digit_model=None
digit_labels=None
model=None
labels=None
net=None
video_path=''
BASE_DIR='web/InputVideos'

def handle_video_option(request):
    global video_path
    print("Inside handle video option")
    if(request.method == 'POST'):
        value=request.POST['VideoTitle']
        if(Videos.objects.filter(title=value).exists()):
            print("Video with entered title exists :)")
            video=Videos.objects.get(title=value)
            video_name_tokens=str(video.video).split('/')
            VideoName=video_name_tokens[-1]
            print("VideoFileName: ",VideoName)
            video_path = BASE_DIR + '/'+VideoName
            if(os.path.exists(video_path) == False):
                return HttpResponse("Video Path does not exists :(")

            if len(video_path) != 0:
                msg=video.title
                return render(request,'web/Dashboard.html',{'option_selected':msg})
            else:
                return HttpResponse("Video Path length is zero(Not Valid).")
        else:
            return HttpResponse("Video corresponding to input title does not exists :(")

    return render(request,'web/play_videos.html',{})

def complete_anpr(request):
    #Deepak's Module
    global video_path

    if((len(video_path) == 0)):
        return HttpResponse("<h2>Error: Please select input video from \"Access Input videos\" section from dashboard :)</h2>")

    if((os.path.exists(video_path) == False)):
        return HttpResponse("Error: Video path set does not exists... :(")

    image_list=MovementDetection(video_path)

    if len(image_list) == 0:
        return HttpResponse("Error: ImageList Empty, Check MovementDetection function :(")

    img=image_list[-1]  #Sending last image from the image list for quality to be max

    #cv2.imwrite('Deepak_img.png',img);  #Testing Deepak's module

    # Work on the below error condition
    # Update: Whenever we make changes while server is running,
    # the server restarts itself, thus net variable and classes are not set to correct values.
    # The existing values of these variables are set to default intialized values as done
    # in the global area section.

    if len(classes) == 0 or net == None:
        return HttpResponse("Error: Empty Classes or net == None")

    flag,coordinates = DetectVehicle(img, net, classes)# Tanmay Module

    if flag == True:
        x,y,w,h=coordinates
        #Avoiding negative coordinates which results in null image further.
        if(x>0 and y>0 and w>0 and h>0):
            img=img[y:y+h,x:x+w]

        print(x, y, w, h)
        new_image=img.copy()
        img=plate_detection(img)#Omkar Module
        #cv2.imwrite("Omkar_img.png",img);
        #img=None #Testing
        if(img.all()== None):
            context={'image':new_image}

            #cv2.imwrite("VehicleImg.jpg",new_image)
            return render(request,"web/ShowVehicleImg.html",{})
            #Do Something

        width = int(img.shape[1] * 3)
        height = int(img.shape[0] * 3)
        dim = (width, height)

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #cv2.imwrite("Atharva_img_resized.png",resized);
        #Atharva Module
        output_string = recognize_char(resized, first_letter_model, first_letter_labels, second_letter_model,
         second_letter_labels, digit_model, digit_labels, model, labels)
        #Now As String is found out, we can set the
        #video path back to its default value for further processing.
        video_path=''
        #User.objects.filter(User_Name=request.POST['username'], User_Pass=request.POST['password']).exists():
        if(Resident.objects.filter(Resident_Vehicle_Number=output_string).exists()):
            return HttpResponse("Welcome resident :)")
        else:
            if(request.method == 'POST'):
                form=VisitorForm(request.POST)
                if(form.is_valid()):
                    #video=Videos(title=request.POST['title'],video=request.FILES['video'], uploaded_date=timezone.now())
                    visitor=Visitor(Visitor_Name=request.POST['Visitor_Name'],Visiting_Resident_Name=request.POST['Visiting_Resident_Name'],
                    Visitor_Contact_Number=request.POST['Visitor_Contact_Number'], Vehicle_Owner_Name=request.POST['Vehicle_Owner_Name'], Vehicle_Type=request.POST['Vehicle_Type'],
                    Visitor_Vehicle_Number=output_string)
                    visitor.save()
                    return render(request,'web/Dashboard.html', {})
            else:
                form=VisitorForm()
                string=output_string

                return render(request, 'web/VisitorForm.html', {'form':form, 'string':string})
    else:
        return HttpResponse("Error: Vehicle Not Detected(Tanmay Module)")

def load_once():
    global classes
    global first_letter_model, first_letter_labels, second_letter_model, second_letter_labels,digit_model, digit_labels, model, labels, net
    classes=[]
    with open("web/ML_Models/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet("web/ML_Models/yolov3.weights", "web/ML_Models/yolov3.cfg")
    json_file = open('web/ML_Models/MobileNets_first_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    first_letter_model = model_from_json(loaded_model_json)

    first_letter_model.load_weights("web/ML_Models/first_letter_dataset.h5")
    print("First Letter Model loaded successfully...")

    first_letter_labels = LabelEncoder()
    first_letter_labels.classes_ = np.load('web/ML_Models/license_first_character_classes.npy')
    print("First Letter Labels loaded successfully...")

    json_file = open('web/ML_Models/MobileNets_second_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    second_letter_model = model_from_json(loaded_model_json)

    second_letter_model.load_weights("web/ML_Models/second_letter_dataset.h5")
    print("Second Letter Model loaded successfully...")

    second_letter_labels = LabelEncoder()
    second_letter_labels.classes_ = np.load('web/ML_Models/license_second_character_classes.npy')
    print("Second Letter Labels loaded successfully...")

    json_file = open('web/ML_Models/MobileNets_digit_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    digit_model = model_from_json(loaded_model_json)

    digit_model.load_weights("web/ML_Models/digits_dataset.h5")
    print("Digit Model loaded successfully...")

    digit_labels = LabelEncoder()
    digit_labels.classes_ = np.load('web/ML_Models/license_digit_classes.npy')
    print("Digit Labels loaded successfully...")

    json_file = open('web/ML_Models/MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("web/ML_Models/License_character_recognition.h5")
    print("Model loaded successfully...")

    labels = LabelEncoder()
    labels.classes_ = np.load('web/ML_Models/license_character_classes.npy')
    print("Labels loaded successfully...")



def fill_visitor_form(request):
    if(request.method == 'POST'):
        form=VisitorForm(request.POST)
        if(form.is_valid()):
            #video=Videos(title=request.POST['title'],video=request.FILES['video'], uploaded_date=timezone.now())
            visitor=Visitor(Visitor_Name=request.POST['Visitor_Name'],Visiting_Resident_Name=request.POST['Visiting_Resident_Name'],
            Visitor_Contact_Number=request.POST['Visitor_Contact_Number'], Vehicle_Owner_Name=request.POST['Vehicle_Owner_Name'], Vehicle_Type=request.POST['Vehicle_Type'],
            Visitor_Vehicle_Number=output_string)
            visitor.save()
            return render(request,'web/Dashboard.html', {})
    else:
        form=VisitorForm()
        string=output_string

        return render(request, 'web/VisitorForm.html', {'form':form, 'string':string})

#View to be executed when number plate is manually entered
def check_vehicle_number(request):
    output_string=request.GET['username']
    print(output_string)
    if(Resident.objects.filter(Resident_Vehicle_Number=output_string).exists()):
        return HttpResponse("Welcome resident :)")
    else:
        if(request.method == 'POST'):
            form=VisitorForm(request.POST)
            if(form.is_valid()):
                #video=Videos(title=request.POST['title'],video=request.FILES['video'], uploaded_date=timezone.now())
                visitor=Visitor(Visitor_Name=request.POST['Visitor_Name'],Visiting_Resident_Name=request.POST['Visiting_Resident_Name'],
                Visitor_Contact_Number=request.POST['Visitor_Contact_Number'], Vehicle_Owner_Name=request.POST['Vehicle_Owner_Name'], Vehicle_Type=request.POST['Vehicle_Type'],
                Visitor_Vehicle_Number=output_string)
                visitor.save()
                return render(request,'web/Dashboard.html', {})
        else:
            form=VisitorForm()
            string=output_string

            return render(request, 'web/VisitorForm.html', {'form':form, 'string':string})

# Route to dashboard page post-login if credentials are valid
def dashboard(request):
    if(check_session(request) == False):
        return redirect('login')
    return render(request, "web/Dashboard.html",{})

#Check user session authorization
def check_session(request):
    if(request.session.has_key('User_Name')):
       return True
    return False

#View to redirect to sucess page if mail is send successfully.
def mail_transfer_success(request):
    if(check_session(request) == False):
        return redirect('login')
    return render(request, 'web/MailTransferSucess.html',{})

#Function to send mail
def send_mail(request):
    #Validation
    if(check_session(request) == False):
        return redirect('login')

    if(request.method == 'POST'):
        form=EmailForm(request.POST)
        if(form.is_valid()):
            subject=request.POST['subject']
            message_body=request.POST['message_body']
            if(len(EMAIL_HOST_USER) == 0):
                return HttpResponse("EMAIL_HOST_USER not set :/")

            from_email=EMAIL_HOST_USER
            to_mail=[]
            to_mail.append(request.POST['to_mail'])
            email=EmailMessage(subject, message_body, from_email, to_mail)
            email.attach_file('web/logfiles/log_file.txt', mimetype='text/txt')
            try:
                email.send(fail_silently=False,)
            except BadHeaderError:
                return HttpResponse('Invalid Input Header')
            return HttpResponseRedirect('MailTransferSucess')
        else:
            return HttpResponse("Make sure all fields are entered valid")
    else:
        form=EmailForm()
    return render(request,'web/SendEmail.html',{'form':form})

#Uploading video clip for image processing
def upload_video(request):
    if(check_session(request) == False):
        return redirect('login')
    if(request.method == 'POST'):
        form = VideoForm(request.POST, request.FILES)
        if(form.is_valid()):
            video=Videos(title=request.POST['title'],video=request.FILES['video'], uploaded_date=timezone.now())
            video.save()
            return redirect('dashboard')
    else:
        form=VideoForm()
    return render(request,'web/upload.html', {'form':form})

#Render the videos on the webpage
def play_video(request):
    if(check_session(request) == False):
        return redirect('login')
    videos=Videos.objects.all()
    if(videos == None):
        return HttpResponse("No vides found")
    return render(request, 'web/play_videos.html', {'videos':videos})

def loading(request):
    return render(request, 'web/loading.html',{})
#Create your views here.
def login(request):
    login_page=True
    if request.method == 'POST':
        if User.objects.filter(User_Name=request.POST['username'], User_Pass=request.POST['password']).exists():
            user = User.objects.get(User_Name=request.POST['username'], User_Pass=request.POST['password'])
            print("Session Created")
            request.session['User_Name']=user.User_Name
            name=user.User_Name
            context={'user':user,'login_page':login_page, 'name':name}
            load_once()
            return render(request,'web/Dashboard.html', context)
        else:
            request.session['User_Name']=None
            context = {'msg':'Invalid username or password','login_page':login_page}
            return render(request, 'web/login.html',context)

    elif request.method == 'GET':
        if 'action' in request.GET:
            action = request.GET['action']
            if(action == 'logout'):
                if(request.session.has_key('User_Name')):
                    request.session.flush()
                    print("Session flushed")
                return redirect('login')
    return render(request,'web/login.html',{'login_page':login_page})

#View to route to the page resident
def manage_resident(request):
    if(check_session(request) == False):
        return redirect('login')
    return render(request, 'web/manage_resident.html',{})

#Operations to perform on visitors
def manage_visitor(request):
    if(check_session(request) == False):
        return redirect('login')
    return render(request, 'web/manage_visitor.html',{})

#Function to view vistiors curently present in database
def view_visitors(request):
    if(check_session(request) == False):
        return redirect('login')
    post=Visitor.objects.all()
    return render(request, 'web/visitor_info.html',{'post':post})

#Function to add resident in the database
def add_resident(request):
    if(check_session(request) == False):
        return redirect('login')
    if(request.method == 'POST'):
        form = ResidentForm(request.POST)
        if(form.is_valid()):
            form.save()
            return redirect('manage_resident')
    else:
        form=ResidentForm()
    return render(request,'web/add_resident.html',{'form':form})

#Updating existing model in the database
def change_resident_data(request):
    if(check_session(request) == False):
        return redirect('login')
    user_id = request.POST.get('Resident_Vehicle_Key')
    resident = Resident.objects.filter(Resident_Vehicle_Key=user_id).first()
    if(request.method == 'POST'):
        form=UpdateForm(request.POST, instance=resident)
        if(form.is_valid()):
            if Resident.objects.filter(pk=request.POST['Resident_Vehicle_Key']).exists():
                resident_to_update=Resident.objects.get(Resident_Vehicle_Key=request.POST['Resident_Vehicle_Key'])
                resident_to_update.Resident_Name=request.POST['Resident_Name']
                resident_to_update.House_Number=request.POST['House_Number']
                resident_to_update.Resident_Vehicle_Number=request.POST['Resident_Vehicle_Number']
                resident_to_update.save()
                return redirect('manage_resident')
            else:
                return HttpResponse("Resident Not Found")
        else:
            return HttpResponse("Form Not Valid")
    else:
        form=UpdateForm()
    return render(request,'web/change_resident_info.html',{'form':form})

#Function to update_resident
def update_resident(request):
    if(check_session(request) == False):
        return redirect('login')
    user_id = request.POST.get('Resident_Vehicle_Key')
    resident = Resident.objects.filter(Resident_Vehicle_Key=user_id).first()
    if(request.method == 'POST'):
        form=RemoveForm(request.POST, instance=resident)
        if(form.is_valid()):
            if Resident.objects.filter(pk=request.POST['Resident_Vehicle_Key']).exists():
                resident_to_update=Resident.objects.get(pk=request.POST['Resident_Vehicle_Key'])
                form=UpdateForm()
                return render(request, 'web/change_resident_info.html', {'form':form})
            else:
                return HttpResponse("Resident Not found")
    else:
        form=RemoveForm()
    return render(request,'web/update_resident.html',{'form':form})

#Removing resident from the database
def remove_resident(request):
    if(check_session(request) == False):
        return redirect('login')

    user_id=request.POST.get('Resident_Vehicle_Key')
    resident = Resident.objects.filter(Resident_Vehicle_Key=user_id).first()
    if(request.method == 'POST'):
        form = RemoveForm(request.POST, instance=resident)
        if(form.is_valid()):
            if(Resident.objects.filter(Resident_Vehicle_Key=request.POST['Resident_Vehicle_Key']).exists()):
                resident_to_delete=Resident.objects.get(Resident_Vehicle_Key=request.POST['Resident_Vehicle_Key'])
                resident_to_delete.delete()
                return redirect('manage_resident')
            else:
                return HttpResponse("Resident Not Found")
    else:
        form=RemoveForm()
    return render(request,'web/remove_resident.html',{'form':form})

#Function to view all objects
def view_resident(request):
    if(check_session(request) == False):
        return redirect('login')
    residents=Resident.objects.all()
    return render(request,'web/view_resident.html',{'residents':residents})

#Function to generate log_file
def generate_log_file(request):
    if(check_session(request) == False):
        return redirect('login')
    visitor_post=Visitor.objects.all()
    now=datetime.now()
    file_object=open('web/logfiles/log_file.txt','w')
    file_object.write("Visitor's Logfile\n")
    file_object.write("Generated at: "+str(now)+'\n')
    for post in visitor_post:
        line='-'*50
        file_object.write(line+'\n')
        file_object.write("Visitor_Name: "+str(post.Visitor_Name)+'\n')
        file_object.write("VisitingResident: "+str(post.Visiting_Resident_Name)+'\n')
        file_object.write("VisitingVehicleNumber: "+str(post.Visitor_Vehicle_Number)+'\n')
    file_object.write(line+'\n')
    file_object.write("END"+'\n')
    file_object.close()
    return render(request,'web/print_data.html',{});
