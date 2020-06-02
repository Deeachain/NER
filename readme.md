## **1.配置训练环境**
ubuntu 16.04<br>
anaconda创建一个python==3.7的虚拟环境
```
pip install -r requirements.txt
```

## 2.NER模型训练和预测
使用kashgari调用bert预训练权重和下游分类任务<br>
本实验可以完成`BiLSTM_CRF_Model`, `CNN_LSTM_Model`, `BiLSTM_Model`, `BiGRU_Model` 或 `BiGRU_CRF_Model`的搭建<br>

训练数据使用的是：source.txt和target.txt<br>
验证数据使用的是：dev.txt和dev-label.txt<br>
测试数据使用的是：test1.txt和test_tgt.txt<br>

运行实验需要替换 BERTEmbedding()中的预训练权重目录，本次实验选用的chinese_L-12_H-768_A-12<br>
/media/ding/Data/model_weight/nlp_weights/tensorflow/google/chinese_L-12_H-768_A-12<br>
权重下载地址[chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

训练模型，主函数修改为：<br>
main(train=True)<br>
设置网络的参数epoch=5;batch_size=64;sequence_length=150;开始训练模型，保存模型到checkpoint文件夹下<br>

测试模型,主函数修改为：<br>
main(predict=True, checkpoint='checkpoint')<br>
开始对模型在测试集上测试，checkpoint为模型保存的文件夹<br>

预测模型，主函数修改为：<br>
main(predict=True, checkpoint='checkpoint')<br>
开始输入一个样本用模型进行预测，checkpoint为模型保存的文件夹<br>


训练模型：<br>
#### ----------BiLSTM_CRF_Model----------
BiLSTM_CRF_Model;epoch=10;batch_size=64;sequence_length=150;<br>
训练方法：
```
python ner.py
```
```
训练过程：
Epoch 1/10
282/282 [==============================] - 172s 610ms/step - loss: 5.8533 - accuracy: 0.9877 - val_loss: 139.6916 - val_accuracy: 0.9901
Epoch 2/10
282/282 [==============================] - 169s 601ms/step - loss: 1.5082 - accuracy: 0.9967 - val_loss: 137.0616 - val_accuracy: 0.9910
Epoch 3/10
282/282 [==============================] - 172s 610ms/step - loss: 1.0474 - accuracy: 0.9975 - val_loss: 134.8197 - val_accuracy: 0.9912
Epoch 4/10
282/282 [==============================] - 165s 585ms/step - loss: 0.7489 - accuracy: 0.9982 - val_loss: 132.7582 - val_accuracy: 0.9917
Epoch 5/10
282/282 [==============================] - 165s 586ms/step - loss: 0.5523 - accuracy: 0.9986 - val_loss: 131.1968 - val_accuracy: 0.9935
Epoch 6/10
282/282 [==============================] - 167s 592ms/step - loss: 0.4096 - accuracy: 0.9990 - val_loss: 129.8139 - val_accuracy: 0.9946
Epoch 7/10
282/282 [==============================] - 164s 582ms/step - loss: 0.3472 - accuracy: 0.9991 - val_loss: 128.5244 - val_accuracy: 0.9946
Epoch 8/10
282/282 [==============================] - 167s 591ms/step - loss: 0.2750 - accuracy: 0.9993 - val_loss: 127.5287 - val_accuracy: 0.9943
Epoch 9/10
282/282 [==============================] - 169s 600ms/step - loss: 0.1996 - accuracy: 0.9995 - val_loss: 126.6077 - val_accuracy: 0.9942
Epoch 10/10
282/282 [==============================] - 165s 584ms/step - loss: 0.1689 - accuracy: 0.9996 - val_loss: 125.9106 - val_accuracy: 0.9942

验证集的测试结果：
           precision    recall  f1-score   support
      PER     0.9415    0.9699    0.9555       166
      ORG     0.8459    0.8543    0.8501       302
      LOC     0.8919    0.8967    0.8943       368
micro avg     0.8853    0.8959    0.8906       836
macro avg     0.8851    0.8959    0.8905       836

测试集的测试结果：
           precision    recall  f1-score   support
      LOC     0.9070    0.9070    0.9070       258
      PER     0.9752    0.9899    0.9825       199
      ORG     0.9020    0.8976    0.8998       205
micro avg     0.9262    0.9290    0.9276       662
macro avg     0.9259    0.9290    0.9275       662

预测：
'十 一 月 ， 我 们 被 称 为 “ 南 京 市 首 届 家 庭 藏 书 状 元 明 星 户 ” 。'
输出：
'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'
```



#### ----------BiGRU_CRF_Model----------
BiGRU_CRF_Model;epoch=10;batch_size=64;sequence_length=150;<br>
训练方法：
```
python ner.py
```

```
Epoch 1/10
282/282 [==============================] - 172s 609ms/step - loss: 6.2045 - accuracy: 0.9876 - val_loss: 147.0721 - val_accuracy: 0.9933
Epoch 2/10
282/282 [==============================] - 163s 578ms/step - loss: 1.3152 - accuracy: 0.9970 - val_loss: 144.6661 - val_accuracy: 0.9921
Epoch 3/10
282/282 [==============================] - 163s 578ms/step - loss: 0.8577 - accuracy: 0.9980 - val_loss: 142.6533 - val_accuracy: 0.9921
Epoch 4/10
282/282 [==============================] - 163s 579ms/step - loss: 0.5895 - accuracy: 0.9986 - val_loss: 140.9434 - val_accuracy: 0.9924
Epoch 5/10
282/282 [==============================] - 164s 583ms/step - loss: 0.4318 - accuracy: 0.9989 - val_loss: 139.7154 - val_accuracy: 0.9923
Epoch 6/10
282/282 [==============================] - 164s 580ms/step - loss: 0.3175 - accuracy: 0.9992 - val_loss: 138.5307 - val_accuracy: 0.9928
Epoch 7/10
282/282 [==============================] - 162s 575ms/step - loss: 0.2626 - accuracy: 0.9993 - val_loss: 137.6368 - val_accuracy: 0.9924
Epoch 8/10
282/282 [==============================] - 162s 576ms/step - loss: 0.2176 - accuracy: 0.9994 - val_loss: 136.7970 - val_accuracy: 0.9921
Epoch 9/10
282/282 [==============================] - 162s 576ms/step - loss: 0.2019 - accuracy: 0.9995 - val_loss: 135.7567 - val_accuracy: 0.9928
Epoch 10/10
282/282 [==============================] - 163s 577ms/step - loss: 0.1632 - accuracy: 0.9996 - val_loss: 135.0596 - val_accuracy: 0.9926
验证集的测试结果：
           precision    recall  f1-score   support
      LOC     0.8909    0.9321    0.9110       368
      PER     0.9583    0.9699    0.9641       166
      ORG     0.8562    0.8874    0.8715       302
micro avg     0.8915    0.9234    0.9072       836
macro avg     0.8918    0.9234    0.9073       836

测试集的测试结果：
           precision    recall  f1-score   support
      LOC     0.9129    0.9341    0.9234       258
      PER     0.9752    0.9899    0.9825       199
      ORG     0.9171    0.9171    0.9171       205
micro avg     0.9329    0.9456    0.9392       662
macro avg     0.9329    0.9456    0.9392       662

预测：
'十 一 月 ， 我 们 被 称 为 “ 南 京 市 首 届 家 庭 藏 书 状 元 明 星 户 ” 。'
输出：
'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'
```

##### 经过实验BiGRU_CRF_Model模型预测指标要高于BiLSTM_CRF_Model，f1-score要高出一个点，原因是GRU只有两个门，LSTM有三个门，参数量减少了1/3，不容易过拟合


## 3.OCR-图片文字识别
采用的是百度API实现,首先安装需要的python-sdk接口<br>
```
cd baidu_api
python setup.py install
```

以下参数是在百度智能云控制台上,用自己账户创建ocr应用生成的,这里用x号注释了<br>[百度智能云](https://login.bce.baidu.com/?account=)

APP_ID = 'xxxxxx'<br>
API_KEY = 'xxxxxx'<br>
SECRET_KEY = 'xxxxxx'<br>

设置需要ocr的图片路径<br>
get_file_content('data/table-pic-for-ocr.png')<br>

运行
```
python ocr.py
```
