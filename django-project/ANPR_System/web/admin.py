from django.contrib import admin
from .models import Resident, Visitor, User, Videos

# Register your models here.
admin.site.register(Resident)
admin.site.register(Visitor)
admin.site.register(User)
admin.site.register(Videos)