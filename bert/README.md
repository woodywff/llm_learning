- code: https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891
- explanation: https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c
- `tokenizer.vocab['[PAD]']`: 0
- `tokenizer.vocab['[CLS]']`: 1  # sos
- `tokenizer.vocab['[SEP]']`: 2  # eos
- `tokenizer.vocab['[MASK]']`: 3  
# 
## Dataset
- the iter of dataset would take a couple of lines from the movie script
- if the two lines are next to each other, the `is_next_label` is 1 otherwise the 2nd sentence may be randomly changed and the `is_next_label` is 0
- for each token, we randomly pick up some places to change them to MASK or any other token,
the corresponding label would be that: the changed place would label the original token id, the other places would be 0
- the two lines would be concatenated as [CLS t1 SEP t2 SEP]
- the two labels would be as [PAD l1 PAD l2 PAD]
- the segment label would be as [1 1 ... 2 2 ...]
- finally we would use PAD to pad lines, labels and segment label

## What makes differences
- It has been trained for two tasks:
  - Masked Language Model
  - Is Next Sentence
- It has mask_x in shape of (batch size, 1, seq len, seq len) other than that of transformer encoder (batch size, 1, 1, seq len),
    it means the mask works for both Q and K
- BERT proved the possibility of pre-training for NLP with transformer architecture just like GPT which is based on decoders.

## How to train a language translation model like transformer
- Download a pre-trained BERT
- Add decoder structures by the end of BERT (of course you need to make some change of the last layers)
- Train it as a transformer