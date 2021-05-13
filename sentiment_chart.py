import base64
from io import BytesIO
from matplotlib.figure import Figure
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# calculates the sentiments of the different files
# the results (compound value) output is in a bar graph
def sentiment_analysis_chart():
	all_ops = open('data/' + 'all_opinions.csv', encoding="utf-8").read().lower()
	strath = open('data/' + 'strath.csv', encoding="utf-8").read().lower()
	uon = open('data/' + 'uon.csv', encoding="utf-8").read().lower()
	usiu = open('data/' + 'usiu.csv', encoding="utf-8").read().lower()
	jkuat = open('data/' + 'jkuat.csv', encoding="utf-8").read().lower()
	other_uni = open('data/' + 'other_uni.csv', encoding="utf-8").read().lower()

	all_score = SentimentIntensityAnalyzer().polarity_scores(all_ops)
	strath_score = SentimentIntensityAnalyzer().polarity_scores(strath)
	uon_score = SentimentIntensityAnalyzer().polarity_scores(uon)
	usiu_score = SentimentIntensityAnalyzer().polarity_scores(usiu)
	jkuat_score = SentimentIntensityAnalyzer().polarity_scores(jkuat)
	other_score = SentimentIntensityAnalyzer().polarity_scores(other_uni)

	all_com = all_score['compound']
	strath_com = strath_score['compound']
	uon_com = uon_score['compound']
	usiu_com = usiu_score['compound']
	jkuat_com = jkuat_score['compound']
	other_com = other_score['compound']

	fig = Figure()
	ax = fig.subplots()
	universities = ['all', 'strath', 'uon', 'usiu', 'jkuat', 'other']
	the_scores = [all_com, strath_com, uon_com, usiu_com, jkuat_com, other_com]
	ax.bar(universities, the_scores)
	buf = BytesIO()
	fig.autofmt_xdate()
	fig.savefig(buf, format="png")

	data = base64.b64encode(buf.getbuffer()).decode("ascii")
	return f"<img src='data:image/png;base64,{data}'/>"

