import string
import base64
from io import BytesIO
from matplotlib.figure import Figure
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords


def emotion_analysis_bar():
	stop_words = stopwords.words('english')

	input_words = open('data/' + 'all_opinions.csv', encoding="utf-8").read().lower()
	cleaned_words = input_words.translate(str.maketrans('', '', string.punctuation))

	processed_opinions = ' '.join([word for word in cleaned_words.split() if word not in stop_words])

	tokenized_opinions = word_tokenize(processed_opinions, "english")
	emotion_list = []
	with open('data/' + 'emotions.txt', 'r') as file:
		for line in file:
			clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
			word, emotion = clear_line.split(':')

			if word in tokenized_opinions:
				emotion_list.append(emotion)
	w = Counter(emotion_list)
	fig = Figure()
	ax = fig.subplots()
	ax.bar(w.keys(), w.values())
	buf = BytesIO()
	fig.autofmt_xdate()
	fig.savefig(buf, format="png")

	data = base64.b64encode(buf.getbuffer()).decode("ascii")
	return f"<img src='data:image/png;base64,{data}'/>"