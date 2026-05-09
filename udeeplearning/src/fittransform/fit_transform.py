from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------------------
# SAMPLE TEXT DATA
# -----------------------------------------

documents = ["free money now", "win free prize", "hello how are you"]

# Create vectorizer
cv = CountVectorizer()

# =====================================================
# 1. FIT
# =====================================================
# Learn vocabulary only
# No matrix is created yet

cv.fit(documents)

print("Vocabulary learned:")
print(cv.vocabulary_)

# Example output:
# {
#   'free': 0,
#   'money': 2,
#   'now': 3,
#   'win': 5,
#   'prize': 4,
#   'hello': 1,
#   ...
# }

# =====================================================
# 2. TRANSFORM
# =====================================================
# Convert text -> vectors
# Uses already learned vocabulary

vectors = cv.transform(documents)

print("\nTransformed matrix:")
print(vectors.toarray())

# Example output:
# [
#   [1 0 1 1 0 0]
#   [1 0 0 0 1 1]
#   [0 1 0 0 0 0]
# ]

# =====================================================
# 3. FIT_TRANSFORM
# =====================================================
# Shortcut:
# fit() + transform() together

cv2 = CountVectorizer()

vectors2 = cv2.fit_transform(documents)

print("\nFit + Transform together:")
print(vectors2.toarray())
