import sys
import string
from nltk import pos_tag

fin = open(sys.argv[1],'r')
fout = open(sys.argv[2],'w')
for line in fin.readlines():
	l = ['(' if s=='-lrb-' else ('[' if s=='-lsb-' else (')' if s=='-rrb-' else (']' if s=='-rsb-' else s))) for s in line.split()]
	p = [(s,'PUNCT') if s in string.punctuation else (s,t) for (s,t) in pos_tag(l)]
	k = [('-lrb-',t) if s=='(' else (('-lsb-',t) if s=='[' else (('-rrb-',t) if s==')' else (('-rsb-',t) if s==']' else (s,t)))) for (s,t) in p]
	for s,t in k:
		fout.write(s+' | '+t+' ')
	fout.write('\n')
fin.close()
fout.close()