from datetime import datetime

from django.shortcuts import render
from django.http import HttpResponse
from .serializers import SudokuSerializer
from .models import Post
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .solver_model.model.evaluate import *
import os, glob

class PostView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        model_path = "/Users/dttai11/Sudoku/solver_model/logs/4"
        ans, validInput, solution = {}, "", ""

        folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/media/post_images/"
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            try:
                solution = sudoku_solver(img_path, model_path)
                validInput = True
                solutionCheck = valiadate_solution(solution)
                ans["solutionCheck"] = solutionCheck
                ans["solution"] = solution
            except:
                validInput = False

        ans["validInput"] = validInput

        Post.objects.all().delete()

        folder = folder + "*"
        list_of_f = glob.glob(folder)
        for i in list_of_f:
            os.remove(i)

        return Response(ans, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        posts_serializer = SudokuSerializer(data=request.data)
        print(request.data['image'])
        if posts_serializer.is_valid():
            posts_serializer.save()
            return Response(posts_serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('error', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)