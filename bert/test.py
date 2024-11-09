import re

text = (
       'Hello, how are you? I am Romeo.n'
       'Hello, Romeo My name is Juliet. Nice to meet you.n'
       'Nice meet you too. How are you today?n'
       'Great. My baseball team won the competition.n'
       'Oh Congratulations, Julietn'
       'Thanks you Romeo'
   )

sentences = re.sub("[.,!?-]", '', text.lower()).split('n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))

print(word_list)
