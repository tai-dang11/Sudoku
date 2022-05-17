from django.db import models

def upload_image(instance, filename):
    # return '/'.join(['covers',filename])
    # return '/'.join(['covers',str(instance.title),filename])
    return '/'.join([filename])

# Create your models here.
class Image(models.Model):
    # title = models.CharField(max_length=8, default="",unique=True)
    cover = models.ImageField(blank=True,null=True,upload_to=upload_image)
