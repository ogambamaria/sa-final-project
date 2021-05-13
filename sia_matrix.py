import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# initialize the Vader model
sia = SentimentIntensityAnalyzer()


# function that calculates the confusion matrix, accuracy, precision, recall and f1 measure.
# Output is the confusion matrix
def sia_accuracy_test():
	with open('data/' + 'strath_testing.csv', encoding="utf-8") as f:
		reader = csv.DictReader(f)
		opinions = list(reader)

		# label the sample dataset
		opinions_df = pd.DataFrame(opinions, columns=['opinion', 'label'])
		opinions_df['label'] = opinions_df['label'].apply(lambda x: 0 if x == '0' else 1)
		opinions_df['prediction'] = opinions_df['opinion'].apply(
			lambda x: 1 if sia.polarity_scores(x)['compound'] >= 0 else 0)

		y_true = opinions_df['label']
		y_pred = opinions_df['prediction']

		cf_matrix = confusion_matrix(y_true, y_pred)

		group_names = ['TP', 'FP', 'FN', 'TN']
		group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
		group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
		all_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]

		all_labels = np.asarray(all_labels).reshape(2,2)

		sns.heatmap(cf_matrix, annot=all_labels, fmt='', cmap='Blues')

		bytes_image = BytesIO()
		plt.savefig(bytes_image, format="png")
		bytes_image.seek(0)
	return bytes_image
