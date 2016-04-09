import os
import sys
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import skimage.io

def prepro_question_answer(imgs):
  '''
    tokenize all questions, answers and multiple choices 
    in the dataset. all punctuations are removed.
  '''

  # preprocess all the questions and answers
  print 'example processed tokens:'
  for i,img in enumerate(imgs):
    img['processed_question_tokens'] = []
    img['processed_answer_tokens'] = []
    img['processed_mc_tokens'] = []
    for j, qa_pair in enumerate(img['qa_pairs']):
      question_txt = str(qa_pair['question']).lower().translate(None, string.punctuation).strip().split()
      img['processed_question_tokens'].append(question_txt)
      answer_txt = str(qa_pair['answer']).lower().translate(None, string.punctuation).strip().split()
      img['processed_answer_tokens'].append(answer_txt)
      processed_mc_tokens = []
      if 'multiple_choices' in qa_pair:
        for mc in qa_pair['multiple_choices']:
          mc_txt = str(mc).lower().translate(None, string.punctuation).strip().split()
          processed_mc_tokens.append(mc_txt)
      img['processed_mc_tokens'].append(processed_mc_tokens)
      if i < 10 and j == 0: print question_txt, answer_txt

def build_vocab(imgs, params):
  '''
    we build a word vocabulary from the questions and answers.
    rare words with frequency lower than a threshold are replaced 
    by a special token UNK (last token in the vocabulary).
  '''
  count_thr = params['word_count_threshold']

  # count up the number of words
  counts = {}
  for img in imgs:
    if img['split'] in ['train', 'val']: # test set shouldn't be used for building vocab
      for txt in img['processed_question_tokens']:
        for w in txt: counts[w] = counts.get(w, 0) + 1
      for txt in img['processed_answer_tokens']:
        for w in txt: counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
  print 'top words and their counts:'
  print '\n'.join(map(str,cw[:20]))

  # print some stats
  total_words = sum(counts.itervalues())
  print 'total words:', total_words
  bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
  vocab = [w for w,n in counts.iteritems() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
  print 'number of words in vocab would be %d' % (len(vocab), )
  print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for img in imgs:
    for txt in img['processed_question_tokens']:
      nw = len(txt)
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    for txt in img['processed_answer_tokens']:
      nw = len(txt)
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print 'max length sentence in raw data: ', max_len
  print 'sentence length distribution (count, number of words):'
  sum_len = sum(sent_lengths.values())
  for i in xrange(max_len+1):
    print '%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len)

  # lets now produce the final annotations
  # additional special UNK token we will use below to map infrequent words to
  print 'inserting the special UNK token'
  vocab.append('UNK')
  
  for img in imgs:
    img['final_questions'] = []
    for txt in img['processed_question_tokens']:
      question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
      img['final_questions'].append(question)
    img['final_answers'] = []
    for txt in img['processed_answer_tokens']:
      answer = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
      img['final_answers'].append(answer)
    img['final_mcs'] = []
    for mc in img['processed_mc_tokens']:
      mcs = []
      for txt in mc:
        mc = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        mcs.append(mc)
      img['final_mcs'].append(mcs)
    
  return vocab

def encode_question_answer(imgs, params, wtoi):
  '''
    encode all questions and answers into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed 
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
  '''

  max_question_length = params['max_question_length']
  max_answer_length = params['max_answer_length']
  MC = params['num_multiple_choice']
  N = len(imgs) # total number of images
  M = sum(len(img['final_answers']) for img in imgs) # total number of QA pairs

  assert M == sum(len(img['final_questions']) for img in imgs), \
    'error: total numbers of questions and answers don\'t match'

  question_label_arrays = []
  answer_label_arrays = []
  mc_label_arrays = []
  question_label_length = np.zeros(M, dtype='uint32')
  answer_label_length = np.zeros(M, dtype='uint32')
  mc_label_length = np.zeros([M, MC], dtype='uint32')
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_id = np.zeros(M, dtype='uint32') # id of the QA pair
  
  question_counter = 0
  answer_counter = 0
  mc_counter = 0
  counter = 1
  for i,img in enumerate(imgs):
    n = len(img['final_questions'])
    assert n > 0, 'error: some image has no QA pairs'

    # getting the labels for questions
    Li = np.zeros((n, max_question_length), dtype='uint32')
    for j,s in enumerate(img['final_questions']):
      question_label_length[question_counter] = min(max_question_length, len(s)) # record the length of this sequence
      label_id[question_counter] = img['qa_pairs'][j]['qa_id']
      question_counter += 1
      for k,w in enumerate(s):
        if k < max_question_length:
          Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and captions are padded with zeros
    question_label_arrays.append(Li)

    # getting the labels for answers
    Li = np.zeros((n, max_answer_length), dtype='uint32')
    for j,s in enumerate(img['final_answers']):
      answer_label_length[answer_counter] = min(max_answer_length, len(s)) # record the length of this sequence
      assert label_id[answer_counter] == img['qa_pairs'][j]['qa_id'], 'order of answers doesn\'t match order of questions'
      answer_counter += 1
      for k,w in enumerate(s):
        if k < max_answer_length:
          Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and QAs are padded with zeros
    answer_label_arrays.append(Li)

    # getting the labels for multiple choices
    Li = np.zeros((n, MC, max_answer_length), dtype='uint32')
    for h,m in enumerate(img['final_mcs']):
      # assert len(m) == MC, 'question has %d multiple choices (expected %d)' % (len(m), MC)
      for j,s in enumerate(m):
        mc_label_length[mc_counter,j] = min(max_answer_length, len(s)) # record the length of this sequence
        for k,w in enumerate(s):
          if k < max_answer_length:
            Li[h,j,k] = wtoi[w]
      mc_counter += 1

    # note: word indices are 1-indexed, and QAs are padded with zeros
    mc_label_arrays.append(Li)
    
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    
    counter += n

  Lq = np.concatenate(question_label_arrays, axis=0)
  La = np.concatenate(answer_label_arrays, axis=0)
  Lmc = np.concatenate(mc_label_arrays, axis=0) # put all the labels together
  assert La.shape[0] == M, 'error: La dimension not matched.'
  assert Lq.shape[0] == M, 'error: Lq dimension not matched.'
  assert Lmc.shape[0] == M, 'error: Lmc dimension not matched.'
  #assert np.all(question_label_length > 0), 'error: some question had no words?'
  #assert np.all(answer_label_length > 0), 'error: some answer had no words?'
  #assert np.all(mc_label_length > 0), 'error: some multiple choices had no words?'

  print 'encoded questions to array of size ', `Lq.shape`
  print 'encoded answers to array of size ', `La.shape`
  print 'encoded multiple choices to array of size ', `Lmc.shape`
  return Lq, La, Lmc, label_start_ix, label_end_ix, question_label_length, answer_label_length, label_id

def load_image(filename, color=True):
  '''
  Load image from file into a numpy array
  -color is the flag for whether to load rgb or grayscale image
  return img as a 3d tensor (HxWx3)
  '''

  img_data = skimage.io.imread(filename, as_grey=not color)
  img = skimage.img_as_float(img_data).astype(np.float32)
  if img.ndim == 2:
    img = img[:, :, np.newaxis]
    if color: img = np.tile(img, (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  return img

def reduce_along_dim(img, dim, weights, indicies): 
  '''
  Perform bilinear interpolation given along the image dimension dim
  -weights are the kernel weights 
  -indicies are the crossponding indicies location
  return img resize along dimension dim
  '''

  other_dim = abs(dim-1)
  if other_dim == 0:  #resizing image width
    weights  = np.tile(weights[np.newaxis,:,:,np.newaxis],(img.shape[other_dim],1,1,3))
    out_img = img[:,indicies,:]*weights
    out_img = np.sum(out_img,axis=2)
  else:   # resize image height     
    weights  = np.tile(weights[:,:,np.newaxis,np.newaxis],(1,1,img.shape[other_dim],3))
    out_img = img[indicies,:,:]*weights
    out_img = np.sum(out_img,axis=1)
  return out_img

def cubic_spline(x):
  '''
  Compute the kernel weights 
  See Keys, "Cubic Convolution Interpolation for Digital Image
  Processing," IEEE Transactions on Acoustics, Speech, and Signal
  Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.
  '''

  absx   = np.abs(x)
  absx2  = absx**2
  absx3  = absx**3 
  kernel_weight = (1.5*absx3 - 2.5*absx2 + 1) * (absx<=1) + (-0.5*absx3 + 2.5* absx2 - 4*absx + 2) * ((1<absx) & (absx<=2))
  return kernel_weight
    
def contribution(in_dim_len , out_dim_len, scale):
  '''
  Compute the weights and indicies of the pixels involved in the cubic interpolation along each dimension.
  
  output:
  weights a list of size 2 (one set of weights for each dimension). Each item is of size OUT_DIM_LEN*Kernel_Width
  indicies a list of size 2(one set of pixel indicies for each dimension) Each item is of size OUT_DIM_LEN*kernel_width
  
  note that if the entire column weights is zero, it gets deleted since those pixels don't contribute to anything
  '''

  kernel_width = 4
  if scale < 1:
    kernel_width = 4 / scale
  x_out = np.array(range(1,out_dim_len+1))  
  #project to the input space dimension
  u = x_out/scale + 0.5*(1-1/scale)
  #position of the left most pixel in each calculation
  l = np.floor( u - kernel_width/2)
  #maxium number of pixels in each computation
  p = int(np.ceil(kernel_width) + 2)
  indicies = np.zeros((l.shape[0],p) , dtype = int)
  indicies[:,0] = l
  for i in range(1,p):
    indicies[:,i] = indicies[:,i-1]+1

  #compute the weights of the vectors
  u = u.reshape((u.shape[0],1))
  u = np.repeat(u,p,axis=1)
  if scale < 1:
    weights = scale*cubic_spline(scale*(indicies-u ))
  else:
    weights = cubic_spline((indicies-u))
  weights_sums = np.sum(weights,1)
  weights = weights/ weights_sums[:, np.newaxis] 
  indicies = indicies - 1
  indicies[indicies<0] = 0
  indicies[indicies>in_dim_len-1] = in_dim_len-1 #clamping the indicies at the ends
  valid_cols = np.all( weights==0 , axis = 0 ) == False #find columns that are not all zeros
  indicies = indicies[:,valid_cols]
  weights = weights[:,valid_cols]
  return weights , indicies

def imresize(img , cropped_width , cropped_height):
  '''
  Function implementing matlab's imresize functionality default behaviour
  Cubic spline interpolation with antialiasing correction when scaling down the image.
  '''

  width_scale  = float(cropped_width)  / img.shape[1]
  height_scale = float(cropped_height) / img.shape[0] 
  order   = np.argsort([height_scale , width_scale])
  scale   = [height_scale , width_scale]
  out_dim = [cropped_height , cropped_width]
  weights  = [0,0]
  indicies = [0,0]
  for i in range(0, 2):
    weights[i] , indicies[i] = contribution(img.shape[ i ],out_dim[i], scale[i])
  for i in range(0, len(order)):
    img = reduce_along_dim(img , order[i] , weights[order[i]] , indicies[order[i]])
  return img

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # default arguments work fine with Visual7W
  parser.add_argument('--dataset_json', default='visual7w-toolkit/datasets/visual7w-telling/dataset.json', help='input dataset json file')
  parser.add_argument('--output_json', default='data/qa_data.json', help='output json file')
  parser.add_argument('--output_h5', default='data/qa_data.h5', help='output h5 file')
  parser.add_argument('--num_multiple_choice', default=3, type=int, help='number of multiple choices of each question.')
  parser.add_argument('--max_question_length', default=15, type=int, help='max length of a question, in number of words. questions longer than this get clipped.')
  parser.add_argument('--max_answer_length', default=5, type=int, help='max length of an answer, in number of words. answers longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--image_dim', default=224, type=int, help='dimension of image after rescale (224 is the input image dimension for VGGNet-16)')
  parser.add_argument('--image_path', default='images/v7w_%s.jpg', help='path template based on image id')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  print 'parsed input parameters:'
  print json.dumps(params, indent=2)

  dataset = json.load(open(params['dataset_json'], 'r'))
  prepro_question_answer(dataset['images'])

  # create the vocab
  vocab = build_vocab(dataset['images'], params)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
  image_id = list(set([x['image_id'] for x in dataset['images']]))

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['word_to_ix'] = wtoi
  json.dump(out, open(params['output_json'], 'w'))
  print 'wrote ', params['output_json']

  # encode answers in large arrays, ready to ship to hdf5 file
  Lq, La, Lmc, label_start_ix, label_end_ix, question_label_length, answer_label_length, label_id = encode_question_answer(dataset['images'], params, wtoi)

  # create output h5 file
  f = h5py.File(params['output_h5'], "w")
  f.create_dataset("question_label", dtype='uint32', data=Lq)
  f.create_dataset("answer_label", dtype='uint32', data=La)
  f.create_dataset("mc_label", dtype='uint32', data=Lmc)
  f.create_dataset("qa_start_ix", dtype='uint32', data=label_start_ix)
  f.create_dataset("qa_end_ix", dtype='uint32', data=label_end_ix)
  f.create_dataset("question_label_length", dtype='uint32', data=question_label_length)
  f.create_dataset("answer_label_length", dtype='uint32', data=answer_label_length)
  f.create_dataset("qa_id", dtype='uint32', data=label_id)

  # loading image dataset
  print 'start to process images into hdf5'
  f.create_dataset("image_id", dtype='uint32', data=image_id)
  img_num = len(image_id)
  img_dim = params['image_dim']
  img_data = f.create_dataset("image_data", (img_num, 3, img_dim, img_dim))
  for k, img_id in enumerate(image_id):
    img_path = params['image_path'] % str(img_id)
    img = load_image(img_path)
    img = imresize(img, img_dim, img_dim)
    img_data[k] = img.transpose(2, 0, 1)
    if k % 500 == 0:
      print 'processed %d / %d images' % (k, img_num)

  f.close()
  print 'wrote ', params['output_h5']
