# Author: Saurav Manchanda (manch043@umn.edu)
# coding: utf-8

# In[1]:


import numpy as np
import os, re, sys, argparse
import copy
import math
import random


# In[2]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


# In[3]:


EPSILON = 0.0000001;




# In[30]:


DEBUG=False;

def usage():
    return """python Code/cawa.py -d<Datapath> -c<Num Classes> -s<Random seed> -a<alpha> -k<kernel_size> 
             -v<standard_deviation> -l<learning rate> -y<layer-nodes> -e<num_eopchs> -b<batches> -p<dropout> 
             -u <Unused topic> -m<Clipping> -f<check> -q<scripts folder> -r<results file>"""

def get_args():
    parser = argparse.ArgumentParser(usage=usage());
    parser.add_argument("-d", '--datapath', type=str, default=None, required = True, help='Path to the folder containing data files.');
    parser.add_argument('-c', '--classes', type=int, default=None, required =True, help='Number of classes.');
    parser.add_argument('-s', '--seed', type=int, default=0, required = False, help='Seed for random initializations.');
    parser.add_argument('-a', '--alpha', type=float, default=0.2, required = False, help='Alpha (Default 0.2).');
    parser.add_argument('-k', '--kernel_size', type=int, default=3, required = False, help='Kernel size for smoothing');
    parser.add_argument('-v', '--standard_deviation', type=float, default=-1, required =False, help='Standard deviation for the gaussian kernel, negative input means simple averaging');
    parser.add_argument('-l', '--learning', type=float, default=0.001, required = False, help='Learning rate (Default 0.001).');
    parser.add_argument('-y', '--nodes', type=int, default=256, required = False, help='Number of nodes in neural network (Default 256).');
    parser.add_argument('-e', '--epoch', type=int, default=100, required = False, help='Num epochs (Default 100).');
    parser.add_argument('-b', '--batch', type=int, default=256, required = False, help='Batch size (Default 256).');
    parser.add_argument('-p', '--dropout', type=float, default=0.5, required = False, help='Dropout probability (Default 0.5).');
    parser.add_argument('-u', '--unused', type=int, default=0, required = False, help='Use unused class (Default 0).');
    parser.add_argument('-m', '--clipping', type=float, default=0.25, required = False, help='Clipping value (Default 0.25).');
    parser.add_argument('-f', '--check', type=int, default=10, required = False, help='Check flag (Default 10).');
    parser.add_argument("-q", '--scripts', type=str, default=None, required = True, help='Path to the folder containing python scripts.');
    parser.add_argument("-r", '--results', type=str, default=None, required = True, help='Path to the results output file.');
    return parser.parse_args();


# In[31]:


Datapath = 'data/cmumovie/'
num_classes = 6;
seed0 = 0;
alpha = 0.2;
kernel_size = 3;
kernel_sd = -1;
learning_rate = 0.001;
hidden_dim = 256;
num_epoch = 100;
batch_size = 256;
dropout = 0.50;
use_null = 0;
clipping_value = 0.25;
check_flag = 10;
ScriptPath = "Code/scripts/";
results_file = "results.csv"

if not DEBUG:
    args = get_args();
    Datapath = args.datapath;
    num_classes = args.classes;
    seed0 = args.seed;
    alpha = args.alpha;
    kernel_size = args.kernel_size;
    kernel_sd = args.standard_deviation;
    learning_rate = args.learning;
    hidden_dim = args.nodes; 
    num_epoch = args.epoch; 
    batch_size = args.batch; 
    dropout = args.dropout; 
    use_null = args.unused; 
    clipping_value = args.clipping; 
    check_flag = args.check; 
    ScriptPath = args.scripts; 
    results_file = args.results; 


print("The arguments provided are as follows: ");
print("--datapath "+str(Datapath))
print("--classes "+str(num_classes))
print("--seed "+str(seed0))
print("--alpha "+str(alpha))
print("--kernel_size "+str(kernel_size))
print("--standard_deviation "+str(kernel_sd))
print("--learning "+str(learning_rate))
print("--nodes "+str(hidden_dim))
print("--epoch "+str(num_epoch))
print("--batch "+str(batch_size))
print("--dropout "+str(dropout))
print("--unused "+str(use_null))
print("--clipping "+str(clipping_value))
print("--check "+str(check_flag))
print("--scripts "+str(ScriptPath))
print("--results "+str(results_file))


# In[6]:


sys.path.insert(0, ScriptPath);
import myAUC
import myF1
import mySOV


# In[7]:


np.random.seed(seed0)
random.seed(seed0)
torch.manual_seed(seed0)


# In[8]:


trfile = '%s/train-data.dat' %Datapath
trlblfile = '%s/train-label.dat' %Datapath
tfile = '%s/test-data.dat' %Datapath
tlblfile = '%s/test-label.dat' %Datapath
vocabfile = '%s/vocabs.txt' %Datapath
sfile = '%s/seg-data.dat' %Datapath
slblfile = '%s/seg-label.dat' %Datapath
segmentation_truth = '%s/seg-truth.dat' %Datapath


svalidfile = '%s/seg-data-valid.dat' %Datapath
slblvalidfile = '%s/seg-label-valid.dat' %Datapath
svalid_truth = '%s/seg-truth-valid.dat' %Datapath


# In[9]:


vocab = {};
with open(vocabfile) as file:
    for line in file:
        p = line.strip().split(',');
        vocab[p[0]] = int(p[1])


# In[10]:


segmentation_ground_truth = open(segmentation_truth)
actual_label_list = [];
while True:
    ln = segmentation_ground_truth.readline()
    if len(ln)==0:
        break
    ln = ln.strip().split();
    actual_label_list.append([int(i) for i in ln]);

segmentation_ground_truth.close()


segmentation_ground_truth = open(svalid_truth)
valid_label_list = [];
while True:
    ln = segmentation_ground_truth.readline()
    if len(ln)==0:
        break
    ln = ln.strip().split();
    valid_label_list.append([int(i) for i in ln]);

segmentation_ground_truth.close()


# In[11]:


train_docs = [];
train_labels = [];
fpin = open(trfile)
fpin_label = open(trlblfile)
while True:
    my_text = [];
    ln = fpin.readline()
    if len(ln)==0:
        break
    sents = re.findall('<[0-9]*?>([0-9 ]*)',ln);
    for sent in sents:
        sent = sent.strip();
        if len(sent) > 0:
            my_text.append(sent);
    ln = fpin_label.readline().strip().split();
    my_label_set = set([int(i) for i,val in enumerate(ln) if val=='1'])
    train_docs.append(my_text);
    train_labels.append(my_label_set);

fpin.close()
fpin_label.close()


# In[12]:


label_count = [0]*num_classes;
for i in train_labels:
    for j in i:
        label_count[j] += 1;
        
num_ex = len(train_labels);
pos_weights = [math.sqrt(float(num_ex)/i) for i in label_count];
# pos_weights = pos_weights/np.mean(pos_weights);
neg_weights = [math.sqrt(float(i)/num_ex) for i in label_count];
neg_weights = neg_weights/np.mean(neg_weights);

# pos_weights = [1.0]*num_classes;
neg_weights = [1.0]*num_classes;


# In[13]:


test_docs_topic = [[] for i in range(num_classes)];
test_docs = [];
test_labels = [];
fpin = open(tfile)
fpin_label = open(tlblfile)
while True:
    my_text = [];
    ln = fpin.readline()
    if len(ln)==0:
        break
    sents = re.findall('<[0-9]*?>([0-9 ]*)',ln);
    for sent in sents:
        sent = sent.strip();
        if len(sent) > 0:
            my_text.append(sent);
            
    ln = fpin_label.readline().strip().split();
    my_label_set = set([int(i) for i,val in enumerate(ln) if val=='1'])
    test_docs.append(my_text);
    test_labels.append(my_label_set);
    for i in my_label_set:
        test_docs_topic[i].append(" ".join(my_text));

fpin.close()
fpin_label.close()


# In[14]:


segment_docs_topic = [[] for i in range(num_classes)];
segment_docs = [];
segment_labels = [];
fpin = open(sfile)
fpin_label = open(slblfile)
while True:
    my_text = [];
    ln = fpin.readline()
    if len(ln)==0:
        break
    sents = re.findall('<[0-9]*?>([0-9 ]*)',ln);
    for sent in sents:
        sent = sent.strip();
        if len(sent) > 0:
            my_text.append(sent);
    ln = fpin_label.readline().strip().split();
    my_label_set = set([int(i) for i,val in enumerate(ln) if val=='1'])
    segment_docs.append(my_text);
    segment_labels.append(my_label_set);
    for i in my_label_set:
        segment_docs_topic[i].append(" ".join(my_text));

fpin.close()
fpin_label.close()


# In[15]:


valid_docs_topic = [[] for i in range(num_classes)];
valid_docs = [];
valid_labels = [];
fpin = open(svalidfile)
fpin_label = open(slblvalidfile)
while True:
    my_text = [];
    ln = fpin.readline()
    if len(ln)==0:
        break
    sents = re.findall('<[0-9]*?>([0-9 ]*)',ln);
    for sent in sents:
        sent = sent.strip();
        if len(sent) > 0:
            my_text.append(sent);
    ln = fpin_label.readline().strip().split();
    my_label_set = set([int(i) for i,val in enumerate(ln) if val=='1'])
    valid_docs.append(my_text);
    valid_labels.append(my_label_set);
    for i in my_label_set:
        valid_docs_topic[i].append(" ".join(my_text));

fpin.close()
fpin_label.close()


# In[16]:


class BatchLoader():
    def __init__(self, docs, labels, vocab, num_classes, batch_size, use_null):
        self.data = [[] for i in range(len(docs))];
        self.vocab = vocab;
        self.num_classes = num_classes;
        self.use_null = use_null;
        for i in range(len(docs)):
            word_list = [];
            sent_len = [];
            for j in range(len(docs[i])):
                temp_sent = [int(k) for k in docs[i][j].split()];
                word_list.extend(temp_sent);
                sent_len.append(len(temp_sent));
            self.data[i].append(i);
            self.data[i].append(word_list);
            self.data[i].append(sent_len);
            self.data[i].append(labels[i]);
        
        self.data = sorted(self.data, key=lambda x: -1*len(x[1]));
            
        self.change_pts = [];
        self.current_len = -1;
        for i in range(len(self.data)):
            if len(self.data[i][1]) != self.current_len:
                self.current_len = len(self.data[i][1]);
                self.change_pts.append(i);
                
        self.change_pts.append(len(self.data[i][1]));
        
        self.batch_size = batch_size;
        self.batch_count = math.ceil(len(self.data)/self.batch_size);
        self.starting_indices = [i*self.batch_size for i in range(self.batch_count)];
        self.current_count = 0;
        self.next_batch = [i for i in range(self.batch_count)];
        random.shuffle(self.next_batch);
        
    def shuffle_slice(self, a, start, stop):
        i = start
        while (i < stop-1):
            idx = random.randrange(i, stop)
            a[i], a[idx] = a[idx], a[i]
            i += 1
    
    def get_next_batch(self):
        if self.current_count == self.batch_count:
            for i in range(len(self.change_pts)-1):
                self.shuffle_slice(self.data, self.change_pts[i], self.change_pts[i+1]);
            random.shuffle(self.next_batch);
            self.current_count = 0;
        batch_words = [];
        batch_projection_matrix = [];
        batch_num_words = [];
        batch_num_sentences = [];
        batch_labels = [];
        ids = [];
        temp_batch = self.data[self.starting_indices[self.next_batch[self.current_count]]:min([self.starting_indices[self.next_batch[self.current_count]]+self.batch_size, len(self.data)])];
        max_words = max([len(i[1]) for i in temp_batch])
        max_sent = max([len(i[2]) for i in temp_batch])
        
        
        for i in range(len(temp_batch)):
            padded_words = np.zeros((max_words), dtype=np.int);
            padded_words[:len(temp_batch[i][1])] = temp_batch[i][1];
            for j in range(len(temp_batch[i][1]), len(padded_words)):
                padded_words[j] = len(self.vocab);
            projection_matrix = np.zeros((max_sent, max_words));
            
            l = 0;
            for j in range(len(temp_batch[i][2])):
                for k in range(temp_batch[i][2][j]):
                    projection_matrix[j,l] = 1.0/temp_batch[i][2][j];
                    l += 1;
            
            ids.append(temp_batch[i][0]);
            
            label_decision = [0 for j in range(self.num_classes + self.use_null)];
            for label in temp_batch[i][3]:
                label_decision[label] = 1;
                
            if self.use_null == 1:
                label_decision[self.num_classes] = 1;
            
            batch_num_words.append(len(temp_batch[i][1]));
            batch_num_sentences.append(len(temp_batch[i][2]));
            batch_words.append(autograd.Variable(torch.LongTensor(padded_words)).unsqueeze(0));
            batch_projection_matrix.append(autograd.Variable(torch.FloatTensor(projection_matrix)).unsqueeze(0));
            batch_labels.append(autograd.Variable(torch.FloatTensor(label_decision)).unsqueeze(0));
            
        self.current_count += 1;
        batch_words = torch.cat(batch_words)
        batch_projection_matrix = torch.cat(batch_projection_matrix)
        batch_num_words = torch.LongTensor(batch_num_words)
        batch_num_sentences = torch.LongTensor(batch_num_sentences)
        batch_labels = torch.cat(batch_labels)
        
        return (ids, batch_words, batch_projection_matrix, batch_num_words, batch_num_sentences, batch_labels); 


# In[17]:


class GaussuanConv1d(nn.Module):
    """ Gaussian conv (usable as blurring filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int
         stride: pool stride, int
         padding: pool padding, int
    """
    def __init__(self, kernel_size=3, stride=1, sig=1.0):
        super(GaussuanConv1d, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = int((kernel_size - 1) / 2)  # convert to l, r, t, b
        self.sig = 1.0;
        self.avg = (kernel_size - 1) / 2;
        self.x_cord = torch.arange(kernel_size)
        self.x_cord = self.x_cord.float()
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        conv = 1 / (self.sig * math.sqrt(2 * math.pi)) * torch.exp(-((self.x_cord - self.avg) / self.sig) ** 2 / 2)
        conv = conv / torch.sum(conv);
        conv = conv.view(1, 1, 1, -1);
        y = F.pad(x, (self.padding, self.padding), "replicate")
        y = y.unfold(y.dim()-1, self.k, self.stride);
        conv = conv.expand(y.size())
        y = y*conv;
        y = y.sum(dim=-1)
        return y


# In[18]:


class sentence_encoder(nn.Module):
    def __init__(self, hidden_size=50, dropout=0.5, vocab_size=1000):
        super(sentence_encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=dropout);
        
        self.key_embedding = nn.Embedding(vocab_size+1, hidden_size, padding_idx=vocab_size);
        self.value_embedding = nn.Embedding(vocab_size+1, hidden_size, padding_idx=vocab_size);
        
    def forward(self, docs, proj_mat, hidden=None):
        """
        Args:
            - docs: (batch_size, max_doc_size)
            - proj_mat: (batch_size, max_sent, max_doc_size)
        Returns:
            - keys: (batch_size, max_sent, hidden_size)
            - values : (batch_size, max_sent, hidden_size)
        """
        
        # (batch_size, max_doc_size) => (batch_size, max_doc_size, hidden_size)
        key_emb = self.dropout(self.key_embedding(docs))
        value_emb = self.dropout(self.value_embedding(docs))
    
        
        # keys: (batch_size, max_sent, hidden_size)
        # values : (batch_size, max_sent, hidden_size)
        keys = torch.bmm(proj_mat, key_emb)
        values = torch.bmm(proj_mat, value_emb)
        
        
        return keys, values


# In[19]:


class class_attention(nn.Module):
    def __init__(self, hidden_size=50, dropout=0.5, num_classes=2, bias=False, kernel_size=3, kernel_sd=-1.0):
        super(class_attention, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout);
        self.hidden_size = hidden_size
        self.num_classes = num_classes;
        self.query_embedding_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.query_embedding_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.query_embedding_3 = nn.Linear(self.hidden_size, self.num_classes, bias=bias)
        if kernel_sd < 0:
            self.avgpool = nn.AvgPool1d(kernel_size, stride=1, padding=int((kernel_size - 1) / 2), count_include_pad=False);
        else:
            self.avgpool = GaussuanConv1d(kernel_size=kernel_size, stride=1, sig=kernel_sd)
        
    def forward(self, sent_keys, sent_values, sent_lens):
        """
        Args:
            - sent_keys: (batch_size, max_doc_size, hidden_size)
            - sent_values: (batch_size, max_doc_size, hidden_size)
            - sent_lens: (batch_size, actual_sent_lens)
        Returns:
            - attention_values: (batch_size, max_doc_size, num_classes)
            - class_inputs: (batch_size, num_classes, hidden_size)
        """
        
        # key-query products
        p1 = self.dropout(torch.tanh(self.query_embedding_1(sent_keys)));
        p2 = self.dropout(torch.tanh(self.query_embedding_2(p1)));
        p3 = self.query_embedding_3(p2 + p1);
        
        kq_prod = p3;
        
        padding_indices = sent_lens.unsqueeze(1).unsqueeze(2).expand(kq_prod.size()) - 1;
        
        break_kq_prod = torch.gather(kq_prod, 1, padding_indices, out=None)
        
        maxlen = kq_prod.size(1)
        idx = torch.arange(maxlen).unsqueeze(0).expand(kq_prod.size()[:2])
        len_expanded = sent_lens.unsqueeze(1).expand(kq_prod.size()[:2])
        mask = idx < len_expanded
        mask = mask.unsqueeze(2).expand(kq_prod.size())
        
        kq_prod = torch.where(mask, kq_prod, break_kq_prod);
        
        kq_prod = kq_prod.transpose(1,2);
        init_size = kq_prod.size();
        kq_prod = kq_prod.contiguous().view(-1, 1, init_size[2]);
        smoothened_kq = self.avgpool(kq_prod);
        smoothened_kq = smoothened_kq.squeeze(1).view(init_size)
        smoothened_kq = smoothened_kq.transpose(1,2);
        attention_values = F.softmax(smoothened_kq, dim=2);

        temp = attention_values.clone();
        temp[~mask] = 0.0
        attention_values = temp;
        class_inputs = torch.bmm(attention_values.transpose(1,2), sent_values);
        return attention_values, class_inputs, smoothened_kq


# In[20]:


class binary_classifier(nn.Module):
    def __init__(self, hidden_size=50, dropout=0.5, bias=True):
        super(binary_classifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout);
        self.input2hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.hidden2hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.hidden2label = nn.Linear(self.hidden_size, 1, bias=bias)  
        
    def forward(self, input_ftr):
        """
        Args:
            - input_ftr: (batch_size, hidden_size)
        Returns:
            - prob: (batch_size)
        """
        
        x = torch.tanh(self.input2hidden(self.dropout(input_ftr)));
        y = torch.tanh(self.hidden2hidden(self.dropout(x)));
        z = self.hidden2label(self.dropout(y+x));
        return z.squeeze(1)


# In[21]:


def masked_attention_loss(logits, target, length, weights):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            normalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, num_classes) depicting label
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each doc in a batch.
        weights: A FloatTensor of size (num_classes)
            which contains the weight of each class.
    Returns:
        loss: An average loss value masked by the length.
    """
    
    log_probs_flat = torch.log(1 + EPSILON - logits)
    maxlen = logits.size(1)
    idx = torch.arange(maxlen).unsqueeze(0).expand(logits.size()[:2])
    len_expanded = length.unsqueeze(1).expand(logits.size()[:2])
    mask = idx < len_expanded
    mask = mask.unsqueeze(2).expand(logits.size())
    
    mask2 = target.unsqueeze(1).expand(logits.size()) > 0
    
    mask = mask * ~mask2
    
    expanded_weights = weights.unsqueeze(0).unsqueeze(0).expand(logits.size());
    expanded_weights = expanded_weights*mask.float();
    
    
    loss = log_probs_flat*expanded_weights;
    
    
    loss = loss.sum()/expanded_weights.sum();
    
    
    return -loss;


# In[22]:


#Load models
s_encoder = sentence_encoder(hidden_size=hidden_dim, vocab_size=len(vocab), dropout=dropout);
attention_model = class_attention(hidden_size=hidden_dim, dropout=dropout, num_classes=num_classes+use_null, bias=True, kernel_size=kernel_size, kernel_sd=kernel_sd);
classifier = [binary_classifier(hidden_size=hidden_dim, dropout=dropout, bias=True) for i in range(num_classes+use_null)];

s_encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, s_encoder.parameters()),lr=learning_rate);
attention_model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, attention_model.parameters()),lr=learning_rate);
classifier_optimizer = [];
for i in range(len(classifier)):
    classifier_optimizer.append(optim.Adam(filter(lambda p: p.requires_grad, classifier[i].parameters()),lr=learning_rate));
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weights))


# In[23]:


def train(words, proj, num_words, num_sents, labels):    
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    
    batch_size = words.size(0)
        
    # -------------------------------------
    # Training mode (enable dropout)
    # -------------------------------------
    s_encoder.train()
    attention_model.train()
    for x in classifier:
        x.train();
    
    # -------------------------------------
    # Zero gradients, since optimizers will accumulate gradients for every backward.
    # -------------------------------------
    s_encoder_optimizer.zero_grad()
    attention_model_optimizer.zero_grad()
    for x in classifier_optimizer:
        x.zero_grad();
        
    # -------------------------------------
    # Forward sentence encoder
    # -------------------------------------
    keys, values = s_encoder(words, proj);
    # -------------------------------------
    # Forward attention
    # -------------------------------------
    attention_values, class_inputs, actual_values = attention_model(keys, values, num_sents)
#     print(actual_values)
    # -------------------------------------
    # Forward classification
    # -------------------------------------
    probs = autograd.Variable(torch.FloatTensor(labels.size()[0], labels.size()[1]))
    for i in range(len(classifier)):
        probs[:,i] = classifier[i](class_inputs[:,i,:]);
        
    # -------------------------------------
    # Compute loss
    # -------------------------------------
    prediction_loss = criterion(probs, labels)
    attention_loss = masked_attention_loss(attention_values, labels, num_sents, torch.FloatTensor(neg_weights));
    
    loss = alpha*prediction_loss + (1-alpha)*attention_loss;
    # -------------------------------------
    # Backward and optimize
    # -------------------------------------
    # Backward to get gradients w.r.t parameters in model.
    loss.backward()
    
    nn.utils.clip_grad_norm_(s_encoder.parameters(), clipping_value);
    nn.utils.clip_grad_norm_(attention_model.parameters(), clipping_value);
    
    for x in classifier:
        nn.utils.clip_grad_norm_(x.parameters(), clipping_value);
    
    # Update parameters with optimizers
    s_encoder_optimizer.step()
    attention_model_optimizer.step()
    for x in classifier_optimizer:
        x.step();
        
    return loss.item()


# In[24]:


def evaluate(words, proj, num_words, num_sents, labels):    
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    
    batch_size = words.size(0)
        
    # -------------------------------------
    # Evaluation mode (disable dropout)
    # -------------------------------------
    s_encoder.eval()
    attention_model.eval()
    for x in classifier:
        x.eval();
    
    # -------------------------------------
    # Zero gradients, since optimizers will accumulate gradients for every backward.
    # -------------------------------------
    s_encoder_optimizer.zero_grad()
    attention_model_optimizer.zero_grad()
    for x in classifier_optimizer:
        x.zero_grad();
        
    # -------------------------------------
    # Forward sentence encoder
    # -------------------------------------
    keys, values = s_encoder(words, proj);
    # -------------------------------------
    # Forward attention
    # -------------------------------------
    attention_values, class_inputs, actual_values = attention_model(keys, values, num_sents)
    # -------------------------------------
    # Forward classification
    # -------------------------------------
    probs = autograd.Variable(torch.FloatTensor(labels.size()[0], labels.size()[1]))
    for i in range(len(classifier)):
        probs[:,i] = classifier[i](class_inputs[:,i,:]);
        
    # -------------------------------------
    # Compute loss
    # -------------------------------------
    loss = criterion(probs, labels)
    return loss.item(), attention_values, torch.sigmoid(probs)


# In[25]:


def test_stats_class():
    tlbl = np.empty((0,num_classes), int);
    y_pred = np.empty((0,num_classes), float);
    beta_vals = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    
    pred_label_list = [[[] for i in range(len(test_docs))] for j in range(len(beta_vals))];
    actual_label_set_list = [[[] for i in range(len(test_docs))] for j in range(len(beta_vals))];
    pred_label_set_list = [[[] for i in range(len(test_docs))] for j in range(len(beta_vals))];
    batch_loader = BatchLoader(test_docs, test_labels, vocab, num_classes, batch_size, use_null=use_null);
    for batch_idx in range(batch_loader.batch_count):
        ids, batch_words, batch_projection_matrix, batch_num_words, batch_num_sentences, batch_labels = batch_loader.get_next_batch();
        loss_val, attention_values, probs = evaluate(batch_words, batch_projection_matrix, batch_num_words, batch_num_sentences, batch_labels);
        tlbl = np.append(tlbl, np.array(batch_labels.data.numpy()[:,:num_classes]), axis=0)
        y_pred = np.append(y_pred, np.array(probs.data.numpy())[:,:num_classes], axis=0)
        for i in range(len(ids)):
            for j in range(len(beta_vals)):
                temp_dist = beta_vals[j]*attention_values[i,:,:num_classes] + (1-beta_vals[j])*probs[i,:num_classes];
                sent_labels = list(temp_dist.data.max(1)[1].numpy())  # get the index of the max log-probability
                pred_label_list[j][ids[i]] = sent_labels[:batch_num_sentences[i]];
                pred_label_set = set(pred_label_list[j][ids[i]]);
                temp_list = [0]*num_classes;
                for label in pred_label_set:
                    temp_list[label] = 1;
                pred_label_set_list[j][ids[i]] = temp_list;
                actual_label_set_list[j][ids[i]] = list(batch_labels[i,:num_classes].data.numpy());
    ans = [];
    (roc, roc_macro) = myAUC.compute_auc(y_pred, tlbl);
    for j in range(len(beta_vals)):
        (micro, samples, macro, weighted) = myF1.compute_f1(np.array(pred_label_set_list[j]), np.array(actual_label_set_list[j]));
        ans.append([roc, roc_macro, micro[2], samples[2], macro[2], weighted[2]]);
    return ans;


# In[26]:


def valid_stats_seg():
    tlbl = np.empty((0,num_classes), int);
    y_pred = np.empty((0,num_classes), float);
    beta_vals = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    pred_label_list = [[[] for i in range(len(valid_docs))] for j in range(len(beta_vals))];
    actual_label_set_list = [[[] for i in range(len(valid_docs))] for j in range(len(beta_vals))];
    pred_label_set_list = [[[] for i in range(len(valid_docs))] for j in range(len(beta_vals))];
    batch_loader = BatchLoader(valid_docs, valid_labels, vocab, num_classes, batch_size, use_null=use_null);
    for batch_idx in range(batch_loader.batch_count):
        ids, batch_words, batch_projection_matrix, batch_num_words, batch_num_sentences, batch_labels = batch_loader.get_next_batch();
        loss_val, attention_values, probs = evaluate(batch_words, batch_projection_matrix, batch_num_words, batch_num_sentences, batch_labels);
        tlbl = np.append(tlbl, np.array(batch_labels.data.numpy()[:,:num_classes]), axis=0)
        y_pred = np.append(y_pred, np.array(probs.data.numpy())[:,:num_classes], axis=0)
        for i in range(len(ids)):
            for j in range(len(beta_vals)):
                temp_dist = beta_vals[j]*attention_values[i,:,:num_classes] + (1-beta_vals[j])*probs[i,:num_classes];
                sent_labels = list(temp_dist.data.max(1)[1].numpy())  # get the index of the max log-probability
                pred_label_list[j][ids[i]] = sent_labels[:batch_num_sentences[i]];
                pred_label_set = set(pred_label_list[j][ids[i]]);
                temp_list = [0]*num_classes;
                for label in pred_label_set:
                    temp_list[label] = 1;
                pred_label_set_list[j][ids[i]] = temp_list;
                actual_label_set_list[j][ids[i]] = list(batch_labels[i,:num_classes].data.numpy());
                
                
    ans = [];
    for j in range(len(beta_vals)):
        (sov, sov_smooth, accuracy, overload_labels, overload_segments) = mySOV.compute_sov(pred_label_list[j], valid_label_list);
        ans.append([sov, sov_smooth, accuracy]);
    return ans;


# In[27]:


def segment_stats_seg():
    tlbl = np.empty((0,num_classes), int);
    y_pred = np.empty((0,num_classes), float);
    beta_vals = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    pred_label_list = [[[] for i in range(len(segment_docs))] for j in range(len(beta_vals))];
    actual_label_set_list = [[[] for i in range(len(segment_docs))] for j in range(len(beta_vals))];
    pred_label_set_list = [[[] for i in range(len(segment_docs))] for j in range(len(beta_vals))];
    batch_loader = BatchLoader(segment_docs, segment_labels, vocab, num_classes, batch_size, use_null=use_null);
    for batch_idx in range(batch_loader.batch_count):
        ids, batch_words, batch_projection_matrix, batch_num_words, batch_num_sentences, batch_labels = batch_loader.get_next_batch();
        loss_val, attention_values, probs = evaluate(batch_words, batch_projection_matrix, batch_num_words, batch_num_sentences, batch_labels);
        tlbl = np.append(tlbl, np.array(batch_labels.data.numpy()[:,:num_classes]), axis=0)
        y_pred = np.append(y_pred, np.array(probs.data.numpy())[:,:num_classes], axis=0)
        for i in range(len(ids)):
            for j in range(len(beta_vals)):
                temp_dist = beta_vals[j]*attention_values[i,:,:num_classes] + (1-beta_vals[j])*probs[i,:num_classes];
                sent_labels = list(temp_dist.data.max(1)[1].numpy())  # get the index of the max log-probability
                pred_label_list[j][ids[i]] = sent_labels[:batch_num_sentences[i]];
                pred_label_set = set(pred_label_list[j][ids[i]]);
                temp_list = [0]*num_classes;
                for label in pred_label_set:
                    temp_list[label] = 1;
                pred_label_set_list[j][ids[i]] = temp_list;
                actual_label_set_list[j][ids[i]] = list(batch_labels[i,:num_classes].data.numpy());
                
                
    ans = [];
    for j in range(len(beta_vals)):
        (sov, sov_smooth, accuracy, overload_labels, overload_segments) = mySOV.compute_sov(pred_label_list[j], actual_label_list);
        ans.append([sov, sov_smooth, accuracy]);
    return ans;


# In[28]:


beta_vals = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
results_target = open(results_file, 'a')
batch_loader_train = BatchLoader(train_docs, train_labels, vocab, num_classes, batch_size, use_null=use_null);
loss_progression = [];
for epoch in range(num_epoch):
    for batch_idx in range(batch_loader_train.batch_count):
        ids, batch_words, batch_projection_matrix, batch_num_words, batch_num_sentences, batch_labels = batch_loader_train.get_next_batch();
        loss_val = train(batch_words, batch_projection_matrix, batch_num_words, batch_num_sentences, batch_labels);
        loss_progression.append(loss_val);
    if epoch%check_flag == 0:
        test_stats = test_stats_class();
        valid_stats = valid_stats_seg();
        seg_stats = segment_stats_seg();
        out_list = [str(seed0), str(alpha), str(kernel_size), str(kernel_sd), str(learning_rate), str(hidden_dim), str(epoch+1), str(batch_size), str(dropout), str(use_null), str(clipping_value)];
        for j in range(len(beta_vals)):
            results_target.write(",".join(out_list));
            results_target.write(","+str(beta_vals[j])+",");
            results_target.write(",".join([str(x) for x in test_stats[j]])+",");
            results_target.write(",".join([str(x) for x in valid_stats[j]])+",");
            results_target.write(",".join([str(x) for x in seg_stats[j]])+"\n");
    
        
results_target.close();

