from nltk.corpus import brown, stopwords
import operator
import string
import pickle

# Data prepping
# data_file 		  = open("striped_words.pickle",'w')
# striped_brwn 	  = [word.lower().strip(string.punctuation) for word in brown.words()]
# striped_stop_brwn = [word for word in striped_brwn if not(word in stopwords.words('english'))]


# pickle.dump(striped_stop_brwn,data_file)
# data_file.close()
# print(len(striped_brwn))
# print(len(striped_stop_brwn))

# Data Reading
data_file = open("striped_words.pickle",'r')
words 	  = pickle.load(data_file)

# print(words)
word_count = dict()
for word in words:
	if not(word in word_count):
		word_count[word]=1
	else:
		word_count[word]+=1

sorted_word_count = sorted(word_count.items(), key=operator.itemgetter(1))
sorted_word_count.reverse()
sorted_word_count = sorted_word_count[1:]


five_words = sorted_word_count[0:5000]
one_words  = sorted_word_count[0:1000]
all_five_words = [datum[0] for datum in five_words]
all_one_words  = [datum[0] for datum in one_words]
joint_one_words= zip(all_one_words,[0.0]*len(all_one_words))

print(one_words[0])
print(all_one_words[0])
print(words[0])
nwc 	= dict()
pc_dict = dict(joint_one_words)
for word in all_five_words:
	nwc[word] = dict(joint_one_words)

print(all_one_words)
for i in range(len(words)):			# Iterate through all text
	if(words[i] in all_five_words):		# If word within the top 5000 words
		# Iterate though w1, w2, w3, w4
		# w1
		# print(words[i-1])
		if((i-1)>=0 and words[i-1] in all_one_words):
			nwc[words[i]][words[i-1]]+=1

		# w2
		if((i-2)>=0 and words[i-2] in all_one_words):
			nwc[words[i]][words[i-2]]+=1

		# w3
		if((i+1)<len(one_words) and words[i+1] in all_one_words):
			nwc[words[i]][words[i+1]]+=1

		# w4
		if((i+2)<len(one_words) and words[i+2] in all_one_words):
			nwc[words[i]][words[i+2]]+=1
	print(i/float(len(words))*100)

print(nwc)
for datum in pwc:
	total = sum([datum[key] for key in datum])
	for key in datum:
		datum[key] = datum[key]/total
