# question-generation

Look into:

* Sentence folder: for the sentence encoder based model **(Recommended)**
* Paragraph folder: for the sentence and paragraph encoder based model

## Model

* A BiLSTM Encoder for the sentence
* A BiLSTM Encoder for the paragraph
* Combination (Concatenation) of the final states of the two encoders act as the intial state for decoder
* Several smaller tweaks to this
* Uses ONMT library

Implementation: PyTorch
Ideas from: https://arxiv.org/abs/1705.00106

A report of the several different directions taken and the experiments is as follows:

Report: http://www.cse.iitd.ac.in/nlpdemo/ques_gen/report.pdf



## Demo

Demo version has been hosted at: http://www.cse.iitd.ac.in/nlpdemo/ques_gen

## Contributers

@Prakhar0409
@ArkaSaha
