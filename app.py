from sia_matrix import *
from sentiment_chart import *
from emotion_analysis import *

from flask import Flask, render_template, send_file
app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/sentiment_analysis')
def all_opinion():
	input_text = open('data/' + 'all_opinions.csv', encoding="utf-8").read().lower()
	score = SentimentIntensityAnalyzer().polarity_scores(input_text)
	neg = score['neg']*100
	pos = score['pos']*100
	neu = score['neu']*100
	com = score['compound']*100

	return render_template('sentiment_analysis.html', result=str(score), neg_score=neg, pos_score=pos, neu_score=neu, com_score=com)


@app.route('/sentiment_bar')
def sentiment_bar():
	s_bar = sentiment_analysis_chart()
	return s_bar


@app.route('/sentiment_accuracy')
def sentient_accuracy():
	with open('data/' + 'strath_testing.csv', encoding="utf-8") as f:
		reader = csv.DictReader(f)
		opinions = list(reader)

		opinions_df = pd.DataFrame(opinions, columns=['opinion', 'label'])
		opinions_df['label'] = opinions_df['label'].apply(lambda x: 0 if x == '0' else 1)
		opinions_df['prediction'] = opinions_df['opinion'].apply(
			lambda x: 1 if sia.polarity_scores(x)['compound'] >= 0 else 0)

		y_true = opinions_df['label']
		y_pred = opinions_df['prediction']

		cf_matrix = confusion_matrix(y_true, y_pred)
		accuracy = accuracy_score(y_true, y_pred)
		precision = precision_score(y_true, y_pred, average='weighted')
		recall = recall_score(y_true, y_pred, average='weighted')
		f1_measure = f1_score(y_true, y_pred, average='weighted')

		ap = accuracy * 100
		pp = precision * 100
		rp = recall * 100
		f1p = f1_measure * 100
	return render_template('sentiment_accuracy.html', cf_matrix=str(cf_matrix), accuracy=accuracy, precision=precision, recall=recall, f1=f1_measure, ap=ap, pp=pp, rp=rp, f1p=f1p)


@app.route('/sia_heatmap', methods=['GET'])
def sia_matrix():
	bytes_obj = sia_accuracy_test()
	return send_file(bytes_obj, attachment_filename='sia_matrix.png', mimetype='image/png')


@app.route('/emotion_bar')
def emotion_bar():
	w_obj = emotion_analysis_bar()
	return w_obj


if __name__ == '__main__':
	app.run()