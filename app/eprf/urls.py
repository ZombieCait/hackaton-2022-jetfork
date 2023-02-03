from django.urls import path, re_path
from . import views

urlpatterns = [
    path('', views.main_page, name='home'),
    path('producers', views.check_producer, name='producers'),
    path('api/v1/dictions', views.get_detail_sets, name='get_detail_sets'),
    # path('api/category/<int:category_id>', views.get_subcategory_info, name='subcategory_info'), # Необходимо править руками в index.html
    path('api/v1/check/single', views.single_check, name='single_check'),
    # path('api/check/json', views.report_json, name='report_json'),
    # path('api/check/xlsx', views.report_xlsx, name='report_xlsx'),
    path('map', views.get_map, name='map'),
]
