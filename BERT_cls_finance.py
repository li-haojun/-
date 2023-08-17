#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


cd /content/drive/MyDrive/BERT


# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install datasets')


# In[ ]:


from transformers import BertTokenizer
from datasets import load_dataset
from transformers import BertModel
from datasets import load_from_disk
import torch
import torch.nn as nn
from transformers import AdamW


# #**自定义数据集**

# In[ ]:


class myDataset(torch.utils.data.Dataset):
    def __init__(self,split=''):
        self.data = load_from_disk('financial_phrasebank/sentences_50agree')
        self.datasplit = self.data.train_test_split(test_size=0.2)
        self.data = self.datasplit[split]

    #__len__,__getitem__这两个函数必须重载
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        #提供myDataset[i]的重载方法
        sen = self.data[i]['sentence']
        label = self.data[i]['label']
        return sen,label
dtrain = myDataset('train')
dtest = myDataset('test')
len(dtrain),len(dtest)


# 为dataLoader中的map类型函数创建自己的数据预处理函数

# In[ ]:


def collate_fn(dataset):
  datasen = [i[0] for i in dataset]
  datalab = [i[1] for i in dataset]
  device = torch.device('cuda')
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  input = tokenizer.batch_encode_plus(batch_text_or_text_pairs=datasen,padding='max_length',truncation=True,max_length=500,return_tensors='pt')
  input_ids = input['input_ids'].to(device)
  attention_mask = input['attention_mask'].to(device)
  token_type_ids = input['token_type_ids'].to(device)
  label = torch.LongTensor(datalab).to(device)
  return input_ids,attention_mask,token_type_ids,label


# 因为dataLoader中传进去的dataset需要是torch.utils.data.Dataset类或其子类，所以上面就不得不将我们在huggingface上的到的数据集重新定义成Dataset子类

# In[ ]:


train_loader = torch.utils.data.DataLoader(dataset=dtrain,batch_size=20,collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(dataset=dtest,batch_size=20,collate_fn=collate_fn)


# In[ ]:


device = torch.device('cuda')
pretrain_model = BertModel.from_pretrained('bert-base-uncased')


# In[ ]:


pretrain_model


# In[ ]:


for i,param in enumerate(pretrain_model.parameters()):
  param.requires_grad_(False)


# In[ ]:


class myBertcls(nn.Module):
  def __init__(self,premodel=None):
    super(myBertcls,self).__init__()
    self.premodel = premodel
    self.cls = nn.Linear(768,3)

  def forward(self,input_ids,attention_mask,token_type_ids):
    with torch.no_grad():
      output = self.premodel(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    output = self.cls(output.last_hidden_state[:,0])
    return output


# In[ ]:


len(train_loader)


# In[ ]:


model = myBertcls(premodel = pretrain_model).to(device)
criterion = nn.CrossEntropyLoss()
opt = AdamW(model.parameters(),lr=5e-4)


# In[ ]:


for j in range(1):
  for i,(input_ids,attention_mask,token_type_ids,label) in enumerate(train_loader):
    if i ==len(train_loader)-1:
      break
    model.train()
    output = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    loss = criterion(output,label)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i%50 == 0 or i == len(train_loader)-2:
      print('epoch:{},batch_num:{},loss:{}'.format(j,i,loss.item()))


# In[ ]:


def test():
  model.eval()
  cor = 0
  total = 0
  for i,(input_ids,attention_mask,token_type_ids,label) in enumerate(test_loader):
    with torch.no_grad():
      output = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    cor += (output.argmax(dim=1)==label).sum().item()
    total += len(label)
  print('accracy:',cor/total)


# In[ ]:


test()


# In[5]:


get_ipython().system('jupyter nbconvert --to python bert_cls_fin.py')

