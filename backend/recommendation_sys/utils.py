#! -*- coding=utf-8 -*-
import json
import codecs
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def content_data():
	hospitals = json.load(codecs.open('./recommendation_sys/doctor.json', 'r', 'utf-8-sig'))
	hospitals = pd.DataFrame.from_dict(hospitals, orient='index')
	hospitals.reset_index(level=0, inplace=True)

	# tag information
	hospitals["tag_literal"] = hospitals['tag'].apply(lambda x: (' ').join(x))
	count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
	genre_mat = count_vect.fit_transform(hospitals["tag_literal"])
	genre_sim = cosine_similarity(genre_mat, genre_mat)
	genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]

	return hospitals, genre_sim_sorted_ind


def find_sim_hospital(df, sorted_ind, title_name, top_n=10):
	title_hospital = df[df["index"] == title_name]

	title_index = title_hospital.index.values
	similar_indexes = sorted_ind[title_index, :top_n]

	print(similar_indexes)
	similar_indexes = similar_indexes.reshape(-1)
	return df.iloc[similar_indexes].to_html()
