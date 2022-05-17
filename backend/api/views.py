from django.shortcuts import render
from rest_framework import viewsets
from .serializers import ImageSerializer
from .models import Image
from django.http import HttpResponse

# Create your views here.
# def main(request):
#     return HttpResponse("check")

class ImageViewSet(viewsets.ModelViewSet):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer

    def post(selfS,request, *args,**kwargs):
        cover = request.data['cover']
        # title = request.data['title']
        # Image.objects.create(title = title,cover=cover)
        Image.objects.create(cover=cover)
        return HttpResponse({'message':'Image uploaded'},status=200)