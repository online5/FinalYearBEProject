from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',views.login,name='login'),
    path('login/', views.login, name='login_success'),
    path('login/dashboard',views.dashboard, name='dashboard'),
    path('login/manage_resident', views.manage_resident, name='manage_resident'),
    path('login/manage_visitor', views.manage_visitor, name='manage_visitor'),
    path('login/visitor_info', views.view_visitors, name='view_visitors'),
    path('login/generate_log_file', views.generate_log_file, name='generate_log_file'),
    path('login/add_resident',views.add_resident, name='add_resident'),
    path('login/view_resident', views.view_resident, name='view_resident'),
    path('login/remove_resident', views.remove_resident, name='remove_resident'),
    path('login/update_resident', views.update_resident, name='update_resident'),
    path('login/change_resident_info', views.change_resident_data, name='change_resident_data'),
    path('login/upload_video', views.upload_video, name='upload_video'),
    path('login/play_video',views.play_video, name='play_video'),
    path('login/complete_anpr',views.complete_anpr, name='complete_anpr'),
    path('login/send_mail', views.send_mail, name='send_mail'),
    path('login/MailTransferSucess', views.mail_transfer_success, name='mail_transfer_success')
]

urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
