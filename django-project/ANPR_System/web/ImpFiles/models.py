from django.db import models
from django.conf import settings
from django.utils import timezone

class Videos(models.Model):
    title=models.CharField(max_length=50)
    video=models.FileField(upload_to='videos/')
    uploaded_date = models.DateTimeField(blank=True, null=True)
    class Meta:
        verbose_name='video'
        verbose_name_plural='videos'

    def upload(self):
        self.uploaded_date=timezone.now()
        self.save()

    def __str__(self):
        return self.title

# Create your models here.
class User(models.Model):
    User_Name=models.CharField(max_length=50)
    User_Id=models.CharField(max_length=50, primary_key=True)
    User_Pass=models.CharField(max_length=10)
    
    def __str__(self):
        return User_Name

class Visitor(models.Model):
    Visitor_Name=models.CharField(max_length=50)
    Visitor_Vehicle_Number=models.CharField(max_length=50)
    Visiting_Resident_Name=models.CharField(max_length=50)
    Visitor_Contact_Number=models.CharField(max_length=10)
    Vehicle_Owner_Name=models.CharField(max_length=50)
    Vehicle_Type=models.CharField(max_length=50)
    def __str__(self):
        return self.Visitor_Name

class Resident(models.Model):
    Resident_Name=models.CharField(max_length=50)
    House_Number=models.CharField(max_length=100)
    Resident_Vehicle_Number=models.CharField(max_length=15)
    Resident_Vehicle_Key=models.CharField(max_length=10, primary_key=True)

    def __str__(self):
        return self.Resident_Name
