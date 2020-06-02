import kashgari
from kashgari.embeddings import BERTEmbedding
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from kashgari.tasks.labeling import CNN_LSTM_Model, BiLSTM_Model, BiGRU_Model, BiLSTM_CRF_Model, BiGRU_CRF_Model

kashgari.config.use_cudnn_cell = True

'''读取数据'''
# with open("data/dev.txt", "r", encoding='utf-8') as f:
#     data_lines = f.readlines()
# with open("data/dev-lable.txt", "r", encoding='utf-8') as f:
#     label_lines = f.readlines()
#
# data, label = [], []
# for index, sen in enumerate(data_lines):
#     data.append(sen.split())
#     label.append(label_lines[index].split())

'''切分训练集和验证集'''


# train_x, valid_x, train_y, valid_y = train_test_split(data, label, test_size=.3, random_state=0)


def data_load(data_path, label_path):
    with open(data_path, "r", encoding='utf-8') as f:
        data_lines = f.readlines()
    with open(label_path, "r", encoding='utf-8') as f:
        label_lines = f.readlines()

    data, label = [], []
    for index, sen in enumerate(data_lines):
        data.append(sen.split())
        label.append(label_lines[index].split())

    return data, label


'''数据集准备'''
train_x, train_y = data_load(data_path='data/source.txt', label_path='data/target.txt')
valid_x, valid_y = data_load(data_path='data/dev.txt', label_path='data/dev-lable.txt')
test_x, test_y = data_load(data_path='data/test1.txt', label_path='data/test_tgt.txt')


def main(train=False, eval=False, predict=False, checkpoint=None):
    if train:
        bert_embed = BERTEmbedding(
            '/media/ding/Data/model_weight/nlp_weights/tensorflow/google/chinese_L-12_H-768_A-12',
            task=kashgari.LABELING,
            sequence_length=150)

        # 还可以选择 `CNN_LSTM_Model`, `BiLSTM_Model`, `BiGRU_Model` 或 `BiGRU_CRF_Model`
        # model = BiLSTM_CRF_Model(bert_embed)
        model = BiGRU_CRF_Model(bert_embed)
        model.fit(train_x,
                  train_y,
                  x_validate=valid_x,
                  y_validate=valid_y,
                  epochs=10,
                  batch_size=64)

        model.save('checkpoint')

        print('验证集的测试结果：')
        model.evaluate(valid_x, valid_y)  # 验证集输出结果

    if eval:
        model = kashgari.utils.load_model(checkpoint)
        print('测试集的测试结果：')
        model.evaluate(test_x, test_y)  # 测试机输出结果

    if predict:
        model = kashgari.utils.load_model(checkpoint)
        out = model.predict(['十 一 月 ， 我 们 被 称 为 “ 南 京 市 首 届 家 庭 藏 书 状 元 明 星 户 ” 。'.split()])
        print(out)


if __name__ == '__main__':
    main(train=True)
    main(eval=True, checkpoint='checkpoint')
    main(predict=True, checkpoint='checkpoint')
