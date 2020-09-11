#! -*- coding=utf-8 -*-
import json
from django.shortcuts import render
from django.http import HttpResponse
from recommendation_sys.utils import content_data, find_sim_hospital, get_unseen_surprise, recomm_hospital_by_surprise,\
	collaborative_data, recomm_hospital_by_neural
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response


def recommendation_page(request):
	if request.method == "GET":
		return render(
			request,
			'recommendation_sys/recommendation_page.html', {}
		)
	else:
		return HttpResponse("Invalid Access")


@api_view(['POST'])
def recommendation_by_content(request):
	result = dict()
	result['error'] = -1

	hospital_name = request.POST.get('hospital_name')
	hospitals, genre_sim_sorted_ind = content_data()
	similar_hospitals = find_sim_hospital(hospitals, genre_sim_sorted_ind, hospital_name, 10)

	result['similar_hospitals'] = similar_hospitals
	result['error'] = 0
	return Response(result, status=status.HTTP_200_OK)


@api_view(['POST'])
def recommendation_by_collaborative(request):
	result = dict()
	result['error'] = -1

	user_nickname = request.POST.get('user_nickname')
	ratings, hospitals, svdpp = collaborative_data()
	unseen_hospitals = get_unseen_surprise(ratings, hospitals, user_nickname)
	top_preds = recomm_hospital_by_surprise(svdpp, user_nickname, unseen_hospitals, top_n=10)

	result['top_preds'] = top_preds
	result['error'] = 0
	return Response(result, status=status.HTTP_200_OK)


@api_view(['POST'])
def recommendation_by_neural_collabo(request):
	result = dict()
	result['error'] = -1

	user_nickname = request.POST.get('user_nickname')

	top_preds = recomm_hospital_by_neural(user_id=user_nickname)
	
	result['top_preds'] = top_preds
	result['error'] = 0
	return Response(result, status=status.HTTP_200_OK)