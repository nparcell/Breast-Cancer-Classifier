import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from xgboost import XGBClassifier 
from sklearn.svm import SVC 

class classify(): 

	def __init__(self, dataName): 

		# Start by importing data and making dataFrame 
		self.df = pd.read_csv(dataName) 
		self.df = pd.DataFrame(self.df) 

		# Weird thing with question marks. Make "?"'s "0" 
		new = {} 
		for column in self.df.columns: 
			temp = [] 
			for k, element in enumerate(self.df[column]): 
				if element == "?": 
					temp.append(0) 
				else: 
					temp.append(float(element))  
			new[column] = np.array(temp)  
		
		# print(new) 
		self.df = pd.DataFrame(new) 

		for column in self.df.columns: 
			for k, element in enumerate(self.df[column]): 
				if element == "?": 
					# self.df[column][k].replace("?", "0") 
					print("after", k) 

	def split(self): 

		self.X, self.Y = {}, {} 
		for column in self.df.columns: 
			if "Code" not in column:
				if "Class" in column: 
					self.Y[column] = self.df[column] 
				else: 
					self.X[column] = self.df[column]

		# self.X = np.array(self.X).transpose() 
		# self.Y = np.array(self.Y).transpose() 
		self.X, self.Y = pd.DataFrame(self.X), pd.DataFrame(self.Y) 
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = 0.2) 

	def modeling(self, classifier): 

		self.model = classifier
		self.model.fit(self.X_train, self.Y_train) 

	def predict(self): 

		self.Y_train_preds = self.model.predict(self.X_train) 
		self.Y_test_preds = self.model.predict(self.X_test) 

		train_accuracy = accuracy_score(self.Y_train_preds, self.Y_train) 
		test_accuracy = accuracy_score(self.Y_test_preds, self.Y_test) 
		print("Train accuracy (%): ", train_accuracy*100) 
		print("Test accuracy (%): ", test_accuracy*100) 

	
if __name__ == "__main__": 
	a = classify(dataName = "breast-cancer-wisconsin.csv") 
	a.split() 
	print("Classifier: Gradient Boost")
	a.modeling(classifier = XGBClassifier()) 
	a.predict() 

	print("") 
	a.modeling(classifier = SVC()) 
	a.predict() 