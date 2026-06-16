from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resume = "Python SQL Machine Learning"
job = "Python SQL Data Analysis"

cv = CountVectorizer()
matrix = cv.fit_transform([resume, job])

score = cosine_similarity(matrix)[0][1]

print(f"Match Score: {score * 100:.2f}%")
