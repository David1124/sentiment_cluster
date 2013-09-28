#!/usr/bin/env python
#coding=utf8
#Author: Wei Guannan, Dong Tiancheng

import re
import sys
sys.path.append("./jieba")
import jieba
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten

weibo_src = [] # Source Weibos
weibo_seg = [] # Weibos after chinese word segement
all_wsg = [] # ALL Weibo Sentiment Graph

class WSGraph(dict):
	def __init__(self, vs=[], es=[]):
		"""create a new graph, (vs) is a list of vertices; (es) is a list of edges"""
		self.vertex_count = len(vs)
		self.edge_count = len(es)

		for v in vs:
			self.add_vertex(v)

		for e in es:
			self.add_edge(e)

	def add_vertex(self, v):
		"""add (v) to the graph"""
		self[v] = {}
		self.vertex_count += 1

	def add_edge(self, e):
		"""add (e) to the graph by adding an entry in both direction.
		if there is already an edge connecting these vertices, the
		new edge replace it.
		"""
		self.edge_count += 1
		v, w = e;
		self[v][w] = e;
		self[w][v] = e;

	def GCount(self):
		return self.vertex_count + self.edge_count

class Vertex(object):
	def __init__(self, word=''):
		self.word = word

	def __repr__(self):
		return 'Vertex(%s)' % repr(self.word)

	__str__ = __repr__

class Edge(tuple):
	def __new__(cls, e1, e2, interval):
		return tuple.__new__(cls, (e1, e2, interval))

	def __repr__(self):
		return 'Edge (%s, %s, %d)' % (repr(self[0]), repr(self[1]), repr(self[2]))

	__str__ = __repr__

# Max Common SubGraph of SG1[] and SG2[] TODO
def mcs(SG1, SG2):
	return 0

# return a list of sub graphs TODO
def sub_graph(G):
	return 0

def GDistance(G1, G2):
	sub_g1 = sub_graph(G1)
	sub_g2 = sub_graph(G2)
	maxsg =  mcs(sub_g1, sub_g2)
	return (1 - float(maxsg.GCount())/max(G1.GCount(), G2.GCount()))

class WSentiment:
	def __init__(self):
		self.pos_word = ""
		self.neg_word = ""
		self.pos = ""
		self.neg = ""
		self.result = ""

	def add_pos_word(self, w):
		self.pos_word += w + ","

	def add_neg_word(self, w):
		self.neg_word += w + ","
	
	def set_pos(self,_pos):
		self.pos = _pos
	
	def set_neg(self, _neg):
		self.neg = _neg

	def set_res(self, _res):
		self.result = _res

class SToken:
	def __init__(self, _word, _start, _end):
		self.word = _word
		self.start = _start
		self.end = _end
	
	def __repr__(self):
		return "word: %s, start: %s, end: %s" % (self.word, self.start, self.end)

	__str__ = __repr__

def load_sentiment_dict(filename):
	t = open(filename).read().decode("gbk").split("\n")[:-1]
	return map(lambda x: x[:-2], t)

def test_dict(dict, word):
	if word in dict:
		print "T"
	else:
		print "F"

positive = load_sentiment_dict("positive_dict.txt")
negtive = load_sentiment_dict("negtive_dict.txt")

f = open("../data/ios7_weibo_data_130915", "r")
raw = f.read().decode("utf8")
f.close()
rows = raw.split("\n")[1:-1]
for line in rows:
	content = line.split("\"")[7]
	if re.search("ios", content, re.IGNORECASE):
		weibo_src.append(content)

jieba.set_dictionary('jieba/extra_dict/dict.txt.big')
for each in weibo_src:
	token = []
	seg = jieba.tokenize(each)
	for tk in seg:
		token.append(SToken(tk[0], tk[1], tk[2]))
	weibo_seg.append(token)

#for x in weibo_seg:
#	for t in x:
#		print unicode(t)
	
for each in weibo_seg:
	g = WSGraph()
	sentiment_words = []
	for tk in each:
		word = tk[1]
		if word in positive or word in negtive:
			sentiment_words.append(tk)
	
	for x in zip(sentiment_words, sentiment_words[1:]):
		s1, s2 = x
		v1 = Vertex(s1[0])
		v2 = Vertex(s2[0])
		e = Edge(v1, v2, s2[1]-s1[2])
		g.add_vertex(v1)
		g.add_vertex(v2)
		g.add_edge(e)

	all_wsg.append(g)

# K medoids

'''
features = array(map(lambda x: [x.pos-x.neg], all_wsg))
whitened = whiten(features)
center = kmeans(whitened, 3)[0]
result = vq(features, center)
print result

cluster_res = result[0]

#generate for R
c1 = "c1 <- c(" 
c2 = "c2 <- c(" 
c3 = "c3 <- c(" 
for i in range(0, len(cluster_res)):
	if cluster_res[i] == 0:
		c1 += str(sentiment_result[i].result) + ","
	if cluster_res[i] == 1:
		c2 += str(sentiment_result[i].result) + ","
	if cluster_res[i] == 2:
		c3 += str(sentiment_result[i].result) + ","

c1 = c1[:-1] + ")"
c2 = c2[:-1] + ")"
c3 = c3[:-1] + ")"

print c1
print c2
print c3
'''
