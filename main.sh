. config.sh
python $FORMAT_DATA
svm-train -t 0 $LIBSVM_TRAIN_DOCS
svm-predict $LIBSVM_TEST_DOCS $LIBSVM_MODEL $PREDICTION_RESULTS