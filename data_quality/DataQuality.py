import joblib as jb

class DataQuality(object):
	def __init__(self):
		self.title_vec_opt = jb.load("./parameter/title_vec_opt_20210620.pkl.z")

	def apply_tfidfvectorizer(self, df):
		title_bow_val = self.title_vec_opt.transform(df)

		return title_bow_val