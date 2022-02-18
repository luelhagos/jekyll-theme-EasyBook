---
layout: post
title: "Display informations on Tensorboard PyTorch"
date: 2022-02-19 12:01
categories: PyTorch
permalink: /posts/tensorboard
---

```python
# import libraries
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()
writer = SummaryWriter('runs/mnist_tboard')

import warnings
warnings.filterwarnings('ignore')
```

    2022-02-18 23:25:32.307932: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0



```python
train_data = dsets.MNIST(root='mnist_data/', train=True,transform = transforms.ToTensor(), download=True)
test_data = dsets.MNIST(root='mnist_data/', train=False, transform = transforms.ToTensor(),download=True)
```


```python
batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset=train_data,\
                                          batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,\
                                          batch_size=batch_size)
```


```python
# size of the data
print(train_data.train_data.size())
print(train_data.train_labels.size())
```

    torch.Size([60000, 28, 28])
    torch.Size([60000])



```python
digit_numpy = {0: torch.unsqueeze(train_data.train_data[1], dim=0),\
               1:torch.unsqueeze(train_data.train_data[24], dim=0),\
               2:torch.unsqueeze(train_data.train_data[5], dim=0),\
               3:torch.unsqueeze(train_data.train_data[7], dim=0),\
               4:torch.unsqueeze(train_data.train_data[2], dim=0),\
               5:torch.unsqueeze(train_data.train_data[0], dim=0),\
               6:torch.unsqueeze(train_data.train_data[13], dim=0),\
               7:torch.unsqueeze(train_data.train_data[15], dim=0),\
               8:torch.unsqueeze(train_data.train_data[17], dim=0),\
               9:torch.unsqueeze(train_data.train_data[4], dim=0)}
```


```python
# plot one example
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title(f'{train_data.train_labels[0]}');
```


![png](output_6_0.png)



```python
# image on tensorboard
sample = iter(train_loader)
sample_data, sample_targets = sample.next()
img_grid = torchvision.utils.make_grid(sample_data)
writer.add_image('sample_images', img_grid)
```


```python
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.linear1 = nn.Linear(28*28, 64)
        self.linear2 = nn.Linear(64, 10)
        
    def forward(self, x):
        linear1 = F.relu(self.linear1(x))
        out = self.linear2(linear1)
        return out
```


```python
model = MnistNet()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model#.to(device)
```




    MnistNet(
      (linear1): Linear(in_features=784, out_features=64, bias=True)
      (linear2): Linear(in_features=64, out_features=10, bias=True)
    )




```python
# define the loss function
loss_function = nn.CrossEntropyLoss()
```


```python
# define the optimization
optimizer = optim.Adam(model.parameters(), lr=0.01)
```


```python
# train the model
epoch = 1
# training and testing
for epch in range(epoch):  # loop over the dataset multiple times
    running_loss = 0.0 # running loss
    for step,  (x, y) in enumerate(train_loader):
        inputs, labels = x.reshape(-1, 28*28), y#.to(device), y.to(device) # reshape
        outputs = model(inputs) 
        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        
        loss.backward() # backpropagation
        optimizer.step()
        
        val_pred = torch.max(outputs.data, 1)
        inx = torch.where(labels != val_pred[1], 10, labels)
        fls_inx = [i for i in range(len(inx)) if inx[i]==10]
        if fls_inx != []:
            tr_imgs = torchvision.utils.make_grid([x[i] for i in fls_inx])
            fls_imgs = torchvision.utils.make_grid([digit_numpy[val_pred[1][i].item()] for i in fls_inx])
            writer.add_image('False Predicted', fls_imgs, global_step=step)
            writer.add_image('True Labels',  tr_imgs, global_step=step)
        if (step+1) % 1024 == 0:    # print every 1024 mini-batches
            print (f'Epoch [{epch+1}/{epoch}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            writer.add_scalar('Training loss', loss, global_step=step)
            running_loss = 0.0
print('Finished Training')
```

    Epoch [1/1], Step [1024/3750], Loss: 0.3134
    Epoch [1/1], Step [2048/3750], Loss: 0.0643
    Epoch [1/1], Step [3072/3750], Loss: 0.5109
    Finished Training



```python
# build a confusion matrix

y_pred = []
y_true = []
# constant for classes
classes = ('0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9')
plt.figure(figsize = (16, 10))

i = 0
# iterate over test data
for inputs, labels in test_loader:
    inputs = inputs.reshape(-1, 28*28)
    output = model(inputs)
    val_pred = torch.max(outputs.data, 1)
    y_pred.extend(val_pred[1].numpy()) 

    labels = labels.numpy()
    y_true.extend(labels) 

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])
writer.add_figure('Confusion Matrix', sns.heatmap(df_cm, annot=True).get_figure())
```


```python
# the confusion matrix
df_cm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>57</td>
      <td>178</td>
      <td>0</td>
      <td>62</td>
      <td>175</td>
      <td>147</td>
      <td>162</td>
      <td>57</td>
      <td>0</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65</td>
      <td>222</td>
      <td>0</td>
      <td>56</td>
      <td>229</td>
      <td>131</td>
      <td>215</td>
      <td>71</td>
      <td>0</td>
      <td>146</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>188</td>
      <td>0</td>
      <td>59</td>
      <td>189</td>
      <td>133</td>
      <td>191</td>
      <td>65</td>
      <td>0</td>
      <td>138</td>
    </tr>
    <tr>
      <th>3</th>
      <td>69</td>
      <td>204</td>
      <td>0</td>
      <td>64</td>
      <td>186</td>
      <td>124</td>
      <td>184</td>
      <td>67</td>
      <td>0</td>
      <td>112</td>
    </tr>
    <tr>
      <th>4</th>
      <td>64</td>
      <td>167</td>
      <td>0</td>
      <td>65</td>
      <td>183</td>
      <td>123</td>
      <td>190</td>
      <td>67</td>
      <td>0</td>
      <td>123</td>
    </tr>
    <tr>
      <th>5</th>
      <td>52</td>
      <td>182</td>
      <td>0</td>
      <td>66</td>
      <td>162</td>
      <td>118</td>
      <td>184</td>
      <td>38</td>
      <td>0</td>
      <td>90</td>
    </tr>
    <tr>
      <th>6</th>
      <td>75</td>
      <td>182</td>
      <td>0</td>
      <td>63</td>
      <td>177</td>
      <td>116</td>
      <td>166</td>
      <td>62</td>
      <td>0</td>
      <td>117</td>
    </tr>
    <tr>
      <th>7</th>
      <td>56</td>
      <td>190</td>
      <td>0</td>
      <td>70</td>
      <td>195</td>
      <td>123</td>
      <td>212</td>
      <td>69</td>
      <td>0</td>
      <td>113</td>
    </tr>
    <tr>
      <th>8</th>
      <td>54</td>
      <td>179</td>
      <td>0</td>
      <td>49</td>
      <td>188</td>
      <td>116</td>
      <td>195</td>
      <td>55</td>
      <td>0</td>
      <td>138</td>
    </tr>
    <tr>
      <th>9</th>
      <td>64</td>
      <td>183</td>
      <td>0</td>
      <td>71</td>
      <td>191</td>
      <td>119</td>
      <td>176</td>
      <td>74</td>
      <td>0</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
total = sum([df_cm[str(i)].sum() for i in range(10)])
false_pos = sum([(df_cm[str(i)].sum()-df_cm[str(i)][i]) for i in range(10)])
true_pos = sum(df_cm[str(i)][str(i)] for i in range(10))
false_neg = sum([(df_cm.iloc[i].sum()- df_cm.iloc[i][i]) for i in range(10)])
```


```python
# Accuracy
accuracy = true_pos/total

# Precision
precision = true_pos/(true_pos+false_pos)

# Recall
recall = true_pos/(true_pos+false_neg)

# F1 Score
f1_score = 2*((precision*recall)/(precision+recall))
```


```python
# write it as a text to the tensorboard
writer.add_text("Mertics dataframe", \
                f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
```


```python
# show the tensorboard in localhost
!tensorboard --logdir='runs/mnist_tboard/'
```

    2022-02-18 23:26:08.639373: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
    
    NOTE: Using experimental fast data loading logic. To disable, pass
        "--load_fast=false" and report issues on GitHub. More details:
        https://github.com/tensorflow/tensorboard/issues/4784
    
    Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
    TensorBoard 2.8.0 at http://localhost:6006/ (Press CTRL+C to quit)
    ^C



```python

```
