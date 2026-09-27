"""
URL configuration for starfish project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import include, path
from django.views.generic import RedirectView
from rest_framework import routers
from starfish.router import views

router_v1 = routers.DefaultRouter()
router_v1.register(r'users', views.UserViewSet)
router_v1.register(r'groups', views.GroupViewSet)
router_v1.register(r'sites', views.SiteViewSet, basename="site")
router_v1.register(r'projects', views.ProjectViewSet, basename="project")
router_v1.register(r'project-participants',
                   views.ProjectParticipantViewSet, basename="project-participant")
router_v1.register(r'runs',
                   views.RunViewSet, basename="run")
router_v1.register(r'registry', views.ModelRegistryViewSet,
                   basename="registry")
router_v1.register(r'runs-action', views.RunsActionViewSet,
                   basename="runs-action")

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', RedirectView.as_view(url='starfish/api/v1/')),
    path('starfish/api/v1/', include(router_v1.urls)),
    path('starfish/api/v1/runs',
         views.BulkCreateRunAPIView.as_view(), name='runs'),
    path('starfish/api-auth/',
         include('rest_framework.urls', namespace='rest_framework'))
]
