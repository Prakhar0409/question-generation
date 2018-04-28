import sys
from nltk import word_tokenize, sent_tokenize
import os

def process(st):
	tok = word_tokenize(st.lower())
	tok = ['-lrb-' if s=='(' else ('-lsb-' if s=='[' else ('-rrb-' if s==')' else ('-rsb-' if s==']' else s))) for s in tok]
	s = ' '.join(tok)
	return s

par = sys.argv[1]
sent = sys.argv[2]
parin = open(par+'.in','r')
sentin = open(sent+'.in','r')
parout = open(par+'.out','w')
sentout = open(sent+'.out','w')
if os.stat(sent+'.in').st_size==0:
	for s in parin.readlines():
		ss = sent_tokenize(s)
		for st in ss:
			parout.write(process(s)+'\n')
			sentout.write(process(st)+'\n')
else:
	for s in parin.readlines():
		parout.write(process(s)+'\n')
	for s in sentin.readlines():
		sentout.write(process(s)+'\n')
parin.close()
sentin.close()
parout.close()
sentout.close()