import os
import pandas as pd
import pickle
import joblib as jb
from flask import Flask, request

from data_quality.DataQuality import DataQuality
from scipy.sparse import hstack, vstack, csr_matrix

# load model
model_rf = jb.load("./model/rf_20210620.pkl.z")
pipeline = DataQuality()

# instance flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	test_json = request.get_json()# Acesso de dados fornecidos em requisição do tipo POST na chamada da API
	print(len(test_json))

	# collect data
	if test_json:
		# if isinstance(test_json[0], dict): # teste para verificar se é um valor único
		if len(test_json)==1: # teste para verificar se é um valor único
			df_raw = pd.DataFrame(test_json, index=[0])
			print("Até aqui foi")
		else:
			df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
			print("Ou até aqui foi")

	# preparations
	print("\n\nTamanho do dataframe recebido: {}\n".format(df_raw.shape))
	df_title = df_raw['title']
	print("\n\nTamanho do dataframe de títulos: {}\n".format(df_title.shape))
	df_features = df_raw.drop(['title'], axis=1)
	print("\n\nTamanho do dataframe das features: {}\n".format(df_features.shape))

	title_bow_val = pipeline.apply_tfidfvectorizer(df_title)
	Xval_wtitle = hstack([df_features, title_bow_val])
	print("\n\nTamanho do dataframe depois da vetorização: {}\n".format(Xval_wtitle.shape))

	# predictions
	pred = model_rf.predict(Xval_wtitle)
	print("A previsão foi {}".format(pred))

	df_raw['prediction'] = pred
	print("\n\nTamanho do dataframe recebido depois da previsão: {}\n".format(df_raw.shape))
	print("\nAlteração realizada com sucesso!!\n")

	return df_raw.to_json(orient='records')

if __name__ == '__main__':
	port = os.environ.get('PORT', 5000)
	# start flask
	app.run(host='0.0.0.0', port=port)