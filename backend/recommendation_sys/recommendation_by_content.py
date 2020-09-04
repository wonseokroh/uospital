#! -*- encoding:utf-8 -*-
import os
import json
import codecs
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "healthcare_recommendation_system.settings")

hospitals = json.load(codecs.open('doctor.json', 'r', 'utf-8-sig'))
hospitals = pd.DataFrame.from_dict(hospitals, orient='index')
hospitals.reset_index(level=0, inplace=True)

# tag information
hospitals["tag_literal"] = hospitals['tag'].apply(lambda x: (' ').join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
genre_mat = count_vect.fit_transform(hospitals["tag_literal"])
genre_sim = cosine_similarity(genre_mat, genre_mat)
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]


def find_sim_hospital(df, sorted_ind, title_name, top_n=10):
	title_hospital = df[df["index"] == title_name]

	title_index = title_hospital.index.values
	similar_indexes = sorted_ind[title_index, :top_n]

	print(similar_indexes)
	similar_indexes = similar_indexes.reshape(-1)
	return df.iloc[similar_indexes]


if __name__ == "__main__":
	similar_hospitals = find_sim_hospital(hospitals, genre_sim_sorted_ind, '111의원', 10)
	print(similar_hospitals)
