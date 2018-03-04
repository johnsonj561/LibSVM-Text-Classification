## Text Classification With LibSVM Linear Kernel


### Data Set
WebKB containing 2803 training text data and 1396 test data. This data set contains WWW-pages collected from computer science departments of various universities. These web pages are classified into 4 categories: student, faculty, project, and course. (The first term in each line of the data file is the class label.) The data set has been preprocessed with removing stop words and stemming. So you only need to count the word frequency to generate a document-word matrix before you start classification.

### LibSVM Data Format
LibSVM requires data instances to be in the form:  
label idx:value idx:value idx:value ... \n  
[prepare-data.py] constructs this data format by:  
i) combining training/test data into 1 set  
ii) constructing document-tfidf matrix using Scikit-learn's TfidfVectorizer  
iii) splitting data back into training/test sets  
iv) mapping class labels to integer values  
v) writing documents to appropriate output values in the form label term-idx:tfidf term-idx:tfidf ... \n  

### Procedure
1. Define project parameters in config.sh  
FORMAT_DATA:  python script that will convert data into LibSVM format  
TRAIN_DOCS: file containing training data  
TEST_DOCS: file containing test data  
LIBSVM_TRAIN_DOCS: file where LibSVM formatted training data is written to by FORMAT_DATA  
LIBSVM_TEST_DOCS: file where LibSVM formatted test data is written to by FORMAT_DATA  
LIBSVM_MODEL: file containing LibSVM model, by default LibSVM writes model to LIBSVM_TRAIN_DOCS + '.model'  
PREDICTION_RESULTS: file where LibSVM will write the test data predictions  

2. Execute main.sh  
main.sh will utilize the file definitions in config.sh to:  
i) convert data to LibSVM format  
ii) train a SVM classificaton model using training data with linear kernel function  
iii) predict classification labels for the test data using model trained in part ii    
iv) write classification labels to output results file, as defined in config.sh  
v) print classification accuracy to the console  


### Classification Results  
Accuracy = 90.616% (1265/1396) (classification)  

[prepare-data.py]:preprocessing/prepare-data.py