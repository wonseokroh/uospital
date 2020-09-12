#! -*- coding=utf-8 -*-
import json
import codecs
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from surprise import Reader, Dataset, SVD, accuracy, SVDpp
from surprise.model_selection import cross_validate
from tensorflow.keras.models import load_model


def content_data():
	hospitals = json.load(codecs.open('./crawling_data/doctor.json', 'r', 'utf-8-sig'))
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


def collaborative_data():
	hospitals, ratings = load_rating_data()
	ratings = ratings[['user_id', 'item_id', 'rating']]
	hospitals["item_id"] = hospitals["index"]
	# rating_hospitals = pd.merge(ratings, hospitals, on="item_id")
	# ratings_matrix = ratings.pivot_table('rating', index='user_id', columns='item_id')
	reader = Reader(rating_scale=(0.0, 10.0))
	data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

	svd = SVD(n_factors=50, random_state=0)
	# cross_validate(svd, data, measures=["RMSE", "MAE"], cv=3, verbose=True)
	svd.fit(data.build_full_trainset())

	svdpp = SVDpp(n_factors=50, random_state=0)
	# cross_validate(svdpp, data, measures=["RMSE", "MAE"], cv=3, verbose=True)
	svdpp.fit(data.build_full_trainset())

	return ratings, hospitals, svdpp


def get_unseen_surprise(ratings, hospitals, user_id):
	seen_hospitals = ratings[ratings["user_id"] == user_id]['item_id'].tolist()
	total_hospitals = hospitals["item_id"].tolist()

	unseen_hospitals = [hospital for hospital in total_hospitals if hospital not in seen_hospitals]
	print("hospitals", len(seen_hospitals), "Unseen", len(unseen_hospitals), "Seen", len(total_hospitals))
	return unseen_hospitals


def recomm_hospital_by_surprise(algo, user_id, unseen_hospitals, top_n=10):
	predictions = [algo.predict(str(user_id), str(item_id)) for item_id in unseen_hospitals]

	def sortkey_est(pred):
		return pred.est

	predictions.sort(key=sortkey_est, reverse=True)
	top_predictions = predictions[:top_n]

	top_hospital_ids = [str(pred.iid) for pred in top_predictions]
	top_hospital_rating = [pred.est for pred in top_predictions]

	top_hospital_preds = pd.DataFrame(list(zip(top_hospital_ids, top_hospital_rating)), columns=["item_id", "rating"])

	return top_hospital_preds


# load rating and hospital data
def load_rating_data():
	hospitals = json.load(codecs.open('./crawling_data/doctor.json','r', 'utf-8-sig'))
	hospitals = pd.DataFrame.from_dict(hospitals, orient='index')
	hospitals.reset_index(level=0, inplace=True)
	ratings = pd.read_csv("./crawling_data/review.csv")
	return hospitals, ratings


def load_pretrained_model(path="./pretrained/model.h5"):
	return load_model(path)


def _encode_rating(df):
	user_ids = df["user_id"].unique().tolist()
	user2user_encoded = {x: i for i, x in enumerate(user_ids)}
	userencoded2user = {i: x for i, x in enumerate(user_ids)}

	item_ids = df["item_id"].unique().tolist()
	item2item_encoded = {x: i for i, x in enumerate(item_ids)}
	item_encoded2item = {i: x for i, x in enumerate(item_ids)}
	df["user"] = df["user_id"].map(user2user_encoded)
	df["item"] = df["item_id"].map(item2item_encoded)
	df["rating"] = df["rating"].values.astype(np.float32)

	return user2user_encoded, userencoded2user, item2item_encoded, item_encoded2item


def recomm_hospital_by_neural(user_id):
	hospitals, ratings = load_rating_data()

	ratings = ratings[['user_id', 'item_id', 'rating']]
	hospitals["item_id"] = hospitals["index"]

	model = load_model(path="./pretrained/model.h5")
	user2user_encoded, userencoded2user, item2item_encoded, item_encoded2item = _encode_rating(ratings)

	items_watched_by_user = ratings[ratings.user_id == user_id]
	items_not_watched = hospitals[
		~hospitals["item_id"].isin(items_watched_by_user.item_id.values)
	]["item_id"]
	items_not_watched = list(
		set(items_not_watched).intersection(set(item2item_encoded.keys()))
	)
	items_not_watched = [[item2item_encoded.get(x)] for x in items_not_watched]

	user_encoder = user2user_encoded.get(user_id)
	user_item_array = np.hstack(
		([[user_encoder]] * len(items_not_watched), items_not_watched)
	)
	ratings_pred = model.predict([user_item_array[:,0], user_item_array[:,1]]).flatten()
	top_ratings_indices = ratings_pred.argsort()[-10:][::-1]
	recommended_item_ids = [
		item_encoded2item.get(items_not_watched[x][0]) for x in top_ratings_indices
	]

	recommended_items = hospitals[hospitals["item_id"].isin(recommended_item_ids)]

	return recommended_items.to_json(orient="table")
