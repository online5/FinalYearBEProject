from django import forms
from .models import Resident, Videos, Visitor

class EmailForm(forms.Form):

    subject=forms.CharField(max_length=50)
    to_mail=forms.EmailField()
    message_body=forms.CharField(widget=forms.Textarea)


class VideoForm(forms.ModelForm):
    class Meta:
        model=Videos
        fields = ('title', 'video')

class ResidentForm(forms.ModelForm):
    class Meta:
        model=Resident
        fields = ('Resident_Name', 'House_Number', 'Resident_Vehicle_Number', 'Resident_Vehicle_Key')
class VisitorForm(forms.ModelForm):
    class Meta:
        model=Visitor
        fields=('Visitor_Name', 'Visiting_Resident_Name',
        'Visitor_Contact_Number', 'Vehicle_Owner_Name', 'Vehicle_Type')

class RemoveForm(forms.ModelForm):
    class Meta:
        model=Resident
        fields= ('Resident_Vehicle_Key',)

class UpdateForm(forms.ModelForm):

    class Meta:
        model=Resident
        fields = ('Resident_Name', 'House_Number','Resident_Vehicle_Number','Resident_Vehicle_Key')
