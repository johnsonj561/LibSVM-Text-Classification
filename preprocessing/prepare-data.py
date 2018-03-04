import os
from sklearn.feature_extraction.text import TfidfVectorizer


# define input file paths
# input files have already been preprocessed (stop words and stemming)
trainingInput = os.environ['TRAIN_DOCS']
testInput = os.environ['TEST_DOCS']
inputFiles = [trainingInput, testInput]


# define libsvm-formatted file paths
# input data will be converted to libsvm format (label idx:value idx:value idx:value)
# and written to libsvm-formatted file paths
libsvmTrainingPath = os.environ['LIBSVM_TRAIN_DOCS']
libsvmTestPath = os.environ['LIBSVM_TEST_DOCS']


# define new class labels
classMap = {
  "student": 1,
  "faculty": 2,
  "course": 3,
  "project": 4
}


# combine training data and test data into single corpus
# allows generation of tf-idf matrix with complete dictionary
# stores lengths to reference when separating tf-idf matrix into test/train
allDocuments = []
fileDocumentCount = []
for file in inputFiles:
  inputStream = open(file, 'r')
  documents = inputStream.read().split('\n')
  allDocuments += documents
  inputStream.close()
  fileDocumentCount.append(len(documents))


# separate class label from each document
# store class label in separate array
documentClasses = []
corpus = []
for document in allDocuments:
  arr = document.split()
  documentClass = arr.pop(0).strip()
  documentClasses.append(classMap[documentClass])
  corpus.append(' '.join(arr))


# create tf-idf matrix from the content
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)


# for every document
# construct string of the form (label idx:value idx:value idx:value) 
formattedDocuments = []
for idx, row in enumerate(tfidf_matrix.toarray()):
  result = str(documentClasses[idx])
  for idx, value in enumerate(row):
    if value > 0:
      result += ' ' + str(idx) + ':' + str(value)
  formattedDocuments.append(result)


# split formatted documents back into training / test data
trainingLength = fileDocumentCount[0]
testLength = fileDocumentCount[1]
formattedTrainingData = formattedDocuments[0:trainingLength]
formattedTestData = formattedDocuments[trainingLength:]


# write libsvm formatted training data to output file
directory = os.path.dirname(libsvmTrainingPath)
if not os.path.exists(directory):
    os.makedirs(directory)
trainingStream = open(libsvmTrainingPath, 'w')
trainingStream.write('\n'.join(formattedTrainingData))
trainingStream.close()


# write libsvm formatted test data to output file
directory = os.path.dirname(libsvmTestPath)
if not os.path.exists(directory):
    os.makedirs(directory)
testStream = open(libsvmTestPath, 'w')
testStream.write('\n'.join(formattedTestData))
testStream.close()