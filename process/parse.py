import sys
import spacy

def disp(tok):
	if tok.n_lefts==0 and tok.n_rights==0:
		if tok.text=='(':
			return '-lrb-'
		elif tok.text==')':
			return '-rrb-'
		elif tok.text=='[':
			return '-lsb-'
		elif tok.text==']':
			return '-rsb-'
		else:
			return tok.text
	else:
		s = ''
		for c in tok.lefts:
			s = s+'( '+disp(c)+' ) '
		s = s+tok.text+' '
		for c in tok.rights:
			s = s+'( '+disp(c)+' ) '
		return s

nlp = spacy.load('en_core_web_sm')
fin = open(sys.argv[1],'r')
fout = open(sys.argv[2],'w')
for line in fin.readlines():
	l = ['(' if s=='-lrb-' else ('[' if s=='-lsb-' else (')' if s=='-rrb-' else (']' if s=='-rsb-' else s))) for s in line.rstrip().split()]
	doc = nlp(' '.join(l))
	root = [token for token in doc if token.dep_=='ROOT'][0]
	fout.write('( '+disp(root)+')\n')
fin.close()
fout.close()