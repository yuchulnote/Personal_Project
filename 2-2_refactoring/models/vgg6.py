import os, sys
import numpy as np
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
appended_dir = sys.path.append(parent_dir)

from layers.batch_normalization import BatchNormalization
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.dropout import Dropout
from layers.fc_layer import FC_Layer
from losses.softmax import softmax
from losses.cross_entropy_loss import cross_entropy_error
from losses.softmax_with_loss import SoftmaxWithLoss
from activation.relu import ReLU


class VGG6:
    """
    VGG넷을 응용한 custom model : VGG6 신경망 아키텍처를 구현한 클래스.

    이 클래스는 컨볼루션 신경망 아키텍처의 간소화된 버전인 VGG6를 구현하며, BatchNorm을 추가하였다.
    이미지 분류 작업에 사용한다.

    Parameters:
        input_dim (tuple): 입력 데이터의 차원 (채널 수, 높이, 너비).
        conv_param_1 (dict): 첫 번째 컨볼루션 계층의 매개변수.
        conv_param_2 (dict): 두 번째 컨볼루션 계층의 매개변수.
        conv_param_3 (dict): 세 번째 컨볼루션 계층의 매개변수.
        conv_param_4 (dict): 네 번째 컨볼루션 계층의 매개변수.
        hidden_size (int): 은닉층의 크기.
        output_size (int): 출력층의 크기.
    """
    
    def __init__(self, input_dim = {3, 224, 224},
                 conv_param_1 = {'filter_num' : 16, 'filter_size' : 3, 'pad' : 1, 'stride' : 2},
                 conv_param_2 = {'filter_num' : 32, 'filter_size' : 3, 'pad' : 1, 'stride' : 1},
                 conv_param_3 = {'filter_num' : 32, 'filter_size' : 3, 'pad' : 1, 'stride' : 2},
                 conv_param_4 = {'filter_num' : 64, 'filter_size' : 3, 'pad' : 1, 'stride' : 2},
                 hidden_size = 50, output_size = 2):
        
        self.first_flag = True  # 첫 번째 순전파 플래그
        
        # ======= 가중치 초기화를 위한 노드 수 계산 =======
        pre_node_nums = np.array([3*3*3, 16*3*3, 32*3*3, 32*3*3, 64*7*7, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU 사용 시 권장 초깃값

        self.params = {}  # 가중치와 편향을 저장할 딕셔너리
        pre_channel_num = input_dim[0]  # 입력 데이터의 채널 수
        
        # 각 컨볼루션 계층의 가중치와 편향 초기화
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4]):
            self.params['W' + str(idx + 1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
                                                                                       pre_channel_num,
                                                                                       conv_param['filter_size'],
                                                                                       conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
            self.params['gamma' + str(idx + 1)] = 1.0
            self.params['beta' + str(idx + 1)] = 0.0

        # 완전 연결 계층의 가중치와 편향 초기화
        self.params['W5'] = weight_init_scales[4] * np.random.randn(64 * 7 * 7, hidden_size)
        self.params['b5'] = np.zeros(hidden_size)
        self.params['W6'] = weight_init_scales[5] * np.random.randn(hidden_size, output_size)
        self.params['b6'] = np.zeros(output_size)

        # ======= 계층 생성 =======
        self.layers = []

        # 첫 번째 컨볼루션 계층
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(BatchNormalization(self.params['gamma1'], self.params['beta1']))
        self.layers.append(ReLU())

        # 두 번째 컨볼루션 계층
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(BatchNormalization(self.params['gamma2'], self.params['beta2']))
        self.layers.append(ReLU())
        self.layers.append(Pooling(pool_height=2, pool_width=2, stride=2))

        # 세 번째 컨볼루션 계층
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(BatchNormalization(self.params['gamma3'], self.params['beta3']))
        self.layers.append(ReLU())

        # 네 번째 컨볼루션 계층
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(BatchNormalization(self.params['gamma4'], self.params['beta4']))
        self.layers.append(ReLU())
        self.layers.append(Pooling(pool_height=2, pool_width=2, stride=2))

        # 완전 연결 계층 (Fully-Connected Layer)
        self.layers.append(FC_Layer(self.params['W5'], self.params['b5']))
        self.layers.append(ReLU())
        self.layers.append(Dropout(0.5))
        self.layers.append(FC_Layer(self.params['W6'], self.params['b6']))
        self.layers.append(ReLU())
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()
        
        
    def predict(self, x, train_flg=False, first_flg=None):
        """
        모델을 사용하여 예측을 수행하는 메소드.

        Args:
            x (numpy.ndarray): 입력 데이터.
            train_flg (bool): 훈련 모드 여부. 기본값은 False.
            first_flg (bool, optional): 첫 번째 호출 여부. 기본값은 None.

        Returns:
            numpy.ndarray: 모델의 예측 결과.
        """
        for layer in self.layers:
            # 배치 정규화 계층의 경우 훈련 모드에 따라 다른 동작 수행
            if isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
                    
        return x
    
    
    def loss(self, x, t):
        """
        손실 함수를 계산합니다.

        Args:
            x (numpy.ndarray): 입력 데이터.
            t (numpy.ndarray): 정답 레이블.

        Returns:
            float: 계산된 손실 값.
        """
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)
    

    def accuracy(self, x, t, batch_size=32):
        """
        정확도를 계산합니다.

        Args:
            x (numpy.ndarray): 입력 데이터.
            t (numpy.ndarray): 정답 레이블.
            batch_size (int): 배치 크기.

        Returns:
            float: 계산된 정확도.
        """
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]
    

    def gradient(self, x, t):
        """
        그래디언트를 계산합니다.

        Args:
            x (numpy.ndarray): 입력 데이터.
            t (numpy.ndarray): 정답 레이블.

        Returns:
            dict: 가중치와 편향에 대한 그래디언트.
        """
        # 순전파
        self.loss(x, t)
        
        # 역전파
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = self.layers.copy()
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Convolution, FC_Layer)):
                grads['W' + str(i+1)] = layer.dW
                grads['b' + str(i+1)] = layer.db
            if isinstance(layer, BatchNormalization):
                grads['gamma' + str(i+1)] = layer.dgamma
                grads['beta' + str(i+1)] = layer.dbeta

        return grads

    def save_params(self, file_name='params.pkl'):
        """
        모델의 파라미터를 저장합니다.

        Args:
            file_name (str): 저장할 파일 이름.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        print("Training network params saved:", self.params.keys())
        

    def load_params(self, file_name='params.pkl'):
        """
        저장된 모델 파라미터를 불러옵니다.

        Args:
            file_name (str): 불러올 파일 이름.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Convolution, FC_Layer)):
                layer.W = self.params['W' + str(i+1)]
                layer.b = self.params['b' + str(i+1)]
            if isinstance(layer, BatchNormalization):
                layer.gamma = self.params['gamma' + str(i+1)]
                layer.beta = self.params['beta' + str(i+1)]

        print("Params loaded successfully:", self.params.keys())