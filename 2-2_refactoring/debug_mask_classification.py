import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import glob
import sys, os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from collections import OrderedDict
import pickle
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import wandb
import time
import IProgress
import shutil
import random


class ReLU:
    """
    ReLU(Rectified Linear Unit) 활성화 함수를 구현한 클래스.

    ReLU 활성화 함수는 입력 값이 0 이하일 경우 0을 출력하고,
    0보다 클 경우 입력 값을 그대로 출력한다.
    이 클래스는 순전파(forward)와 역전파(backward) 두 가지 메소드가 있다.
    """

    def __init__(self):
        """
        클래스 초기화 메소드.

        Attributes:
            mask (numpy.array): 입력 x에 대한 마스크. x의 요소가 0 이하일 경우 True.
        """
        self.mask = None
    
    def forward(self, x):
        """
        순전파 메소드.

        입력 x에 대해 ReLU 활성화 함수를 적용합니다.

        Parameters:
            x (numpy.array): 입력 배열.

        Returns:
            numpy.array: ReLU 활성화 함수가 적용된 결과 배열.
        """
        self.mask = (x <= 0)  # 0 이하의 값에 대한 마스크 생성
        out = x.copy()  # 입력 x의 복사본 생성
        out[self.mask] = 0  # 마스크가 True인 위치를 0으로 설정
        
        return out
    
    def backward(self, dout):
        """
        역전파 메소드.

        순전파에서 0 이하로 활성화된 뉴런의 그래디언트를 0으로 설정한다.

        Parameters:
            dout (numpy.array): 출력 그래디언트 배열.

        Returns:
            numpy.array: 수정된 입력 그래디언트 배열.
        """
        dout[self.mask] = 0  # 순전파에서 0 이하의 값에 해당하는 그래디언트를 0으로 설정
        dx = dout  # 수정된 그래디언트 배열
        
        return dx
    

def col2im(col, input_shape, filter_height, filter_width, stride=1, pad=0):
    """
    2차원 배열을 원래의 이미지 데이터 형태로 변환하는 함수.

    주어진 2차원 배열을 컨볼루션 연산의 입력 형태로 되돌린다.
    이 함수는 컨볼루션 계층의 역전파에서 사용한다.

    Parameters:
        col (numpy.array): 2차원 배열.
        input_shape (tuple): 원래 입력 데이터의 형태 (배치 크기, 채널 수, 높이, 너비).
        filter_height (int): 필터의 높이.
        filter_width (int): 필터의 너비.
        stride (int, optional): 스트라이드 값. 기본값은 1.
        pad (int, optional): 패딩 값. 기본값은 0.

    Returns:
        numpy.array: 변환된 이미지 데이터.
    """
    # 입력 데이터의 형태 추출 및 출력 데이터 형태 계산
    Number, Channel, Height, Width = input_shape
    out_height = (Height + 2 * pad - filter_height) // stride + 1
    out_width = (Width + 2 * pad - filter_width) // stride + 1

    # 2차원 배열을 원래의 4차원 형태로 변환
    # reshape으로 원래의 6차원으로 변경
    # transpose로 col : [Number, Channel, filter_height, filter_width, out_height, out_width] 원래 형태로 변경
    col = col.reshape(Number, out_height, out_width, Channel, filter_height, filter_width).transpose(0, 3, 4, 5, 1, 2)

    # 결과 이미지를 저장할 배열 초기화
    img = np.zeros((Number, Channel, Height + 2 * pad + stride - 1, Width + 2 * pad + stride - 1))

    # 이미지 데이터 재구성
    for y in range(filter_height):
        y_max = y + stride * out_height
        for x in range(filter_width):
            x_max = x + stride * out_width
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 패딩 제거하여 최종 이미지 추출
    return img[:, :, pad:Height + pad, pad:Width + pad]


def im2col(input_data, filter_height, filter_width, stride=1, pad=0):
    """
    다수의 이미지 데이터를 2차원 배열로 변환(평탄화).

    주어진 입력 데이터에 대해 지정된 필터 크기, 스트라이드, 패딩을 적용하여
    컨볼루션 연산을 수행하기 위해 2차원 배열로 변환한다.
    
    컨볼루션 연산을 단순한 두 행렬 곱셈으로 수행하기 위한 함수.

    Parameters:
        input_data (numpy.array): 4차원 입력 데이터 (배치 크기, 채널 수, 높이, 너비).
        filter_height (int): 필터의 높이.
        filter_width (int): 필터의 너비.
        stride (int, optional): 스트라이드 값. 기본값은 1.
        pad (int, optional): 패딩 값. 기본값은 0.

    Returns:
        numpy.array: 변환된 2차원 배열.
    """
    # 입력 데이터의 차원을 추출
    Number, Channel, Height, Width = input_data.shape

    # 출력 데이터의 높이와 너비 계산
    out_height = (Height + 2 * pad - filter_height) // stride + 1
    out_width = (Width + 2 * pad - filter_width) // stride + 1
    """
    입력 데이터에 패딩 적용, 입력 데이터 : (배치크기, 채널 수, 높이, 너비)    
    배치 크기에 대한 패딩은 없으므로 (0, 0)
    채널 수에 대한 패딩은 없으므로 (0, 0)
    높이와 너비에 패딩을 추가, 각 차원의 양쪽에 추가될 패딩의 크기
    'constant' : 상수값으로 패딩을 채운다는 의미, 기본값은 0
    """
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    # 2차원 배열로 변환할 때 사용할 임시 배열 생성
    col = np.zeros((Number, Channel, filter_height, filter_width, out_height, out_width))
    
    # 필터를 적용하여 2차원 배열 생성
    for y in range(filter_height):
        y_max = y + stride * out_height
        for x in range(filter_width):
            x_max = x + stride * out_width
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    """        
    2차원 배열로 변환
    바뀐 col : [Number, out_height, out_width, Channel, filter_height, filter_width]
    reshape된 2차원 col : [Number * out_height * out_width, Channel * filter_height * filter_width]
    axis=0 -> Number * out_height * out_width -> 입력 데이터의 모든 위치에서 적용될 수 있는 필터 영역
    axis=1 -> Channel * filter_height * filter_width -> 필터의 각 영역 내의 원소들을 평탄화한 형태
    """
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(Number * out_height * out_width, -1)
    
    return col


class BatchNormalization:
    """
    배치 정규화를 구현한 클래스.

    배치 정규화는 레이어의 입력을 정규화하여 학습을 안정화시키고 가속화하는 기법.
    이 클래스는 각 특성에 대해 평균을 0, 분산을 1로 정규화한 후 스케일 및 시프트 변환을 수행한다.

    Attributes:
        gamma (np.array): 스케일 변환 파라미터.
        beta (np.array): 시프트 변환 파라미터.
        momentum (float): 실행 평균 및 분산을 위한 모멘텀 값.
        running_mean (np.array): 실행 평균.
        running_var (np.array): 실행 분산.
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # 입력 데이터 형태 저장 (합성곱 또는 완전 연결)

        # running_mean과 running_var를 None으로 초기화
        self.running_mean = running_mean
        self.running_var = running_var

        # 역전파 시 사용될 중간 데이터 저장
        self.batch_size = None
        self.centered_input = None
        self.normalized_input = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flag=True):
        """
        순전파 메소드. 입력 데이터에 대해 배치 정규화를 수행합니다.

        Parameters:
            x (np.array): 입력 데이터.
            train_flag (bool): 학습 중인지 여부.

        Returns:
            np.array: 배치 정규화된 출력 데이터.
        """
        self.input_shape = x.shape
        print(self.input_shape)
        original_shape = x.shape
        print(original_shape)
        
        # 2차원이 아닌 경우 형태 변환
        if x.ndim != 2:
            Number, Channel, Height, Width = x.shape
            x = x.reshape(Number, -1)
            print(f"4차원 -> 2차원 : {x.shape}")

        # 여기서 running_mean과 running_var 초기화
        if self.running_mean is None or self.running_var is None:
            Dimension = x.shape[1]  # x의 특성 수
            self.running_mean = np.zeros(Dimension)
            self.running_var = np.zeros(Dimension)
        
        out = self.__forward(x, train_flag)
        
        # 결과를 원래 형태로 되돌림
        if out.ndim != 2:
            out = out.reshape(original_shape)
        
        return out

    def __forward(self, x, train_flag):
        # 실행 평균 및 분산 초기화
        if self.running_mean is None:
            Number, Dimension = x.shape
            self.running_mean = np.zeros(Dimension)
            self.running_var = np.zeros(Dimension)

        if train_flag:
            # 훈련 모드
            mu = x.mean(axis=0)
            centered_input = x - mu
            var = np.mean(centered_input**2, axis=0)
            std = np.sqrt(var + 10e-7)
            normalized_input = centered_input / std

            # 역전파를 위한 중간 데이터 저장
            self.batch_size = x.shape[0]
            self.centered_input = centered_input
            self.normalized_input = normalized_input
            self.std = std

            # 실행 평균 및 분산 업데이트
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            # 추론 모드
            centered_input = x - self.running_mean
            normalized_input = centered_input / (np.sqrt(self.running_var + 10e-7))

        # 감마와 베타를 x의 형태에 맞게 조정
        gamma_reshaped = self.gamma.reshape(1, -1)
        beta_reshaped = self.beta.reshape(1, -1)
        
        if x.ndim == 4:  # 입력이 4차원일 경우
            Number, Channel, Height, Width = x.shape
            gamma_reshaped = self.gamma.reshape(1, Channel, 1, 1)
            beta_reshaped = self.beta.reshape(1, Channel, 1, 1)
        else:  # 입력이 2차원일 경우
            D = x.shape[1]
            gamma_reshaped = self.gamma.reshape(1, D)
            beta_reshaped = self.beta.reshape(1, D)
        
        out = gamma_reshaped * normalized_input + beta_reshaped
        
        return out

    def backward(self, dout):
        """
        역전파 메소드. 배치 정규화의 그래디언트를 계산합니다.

        Parameters:
            dout (np.array): 역전파로부터 전달된 그래디언트.

        Returns:
            np.array: 입력 데이터에 대한 그래디언트.
        """
        if dout.ndim != 2:
            Number, Channel, Height, Width = dout.shape
            dout = dout.reshape(Number, -1)

        dx = self.__backward(dout)
        return dx.reshape(*self.input_shape)

    def __backward(self, dout):
        # 그래디언트 계산
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.normalized_input * dout, axis=0)
        dnormalized_input = self.gamma * dout
        dcentered_input = dnormalized_input / self.std
        dstd = -np.sum((dnormalized_input * self.centered_input) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dcentered_input += (2.0 / self.batch_size) * self.centered_input * dvar
        dmu = np.sum(dcentered_input, axis=0)
        dx = dcentered_input - dmu / self.batch_size

        # 그래디언트 저장
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
    

class Convolution:
    """
    합성곱 계층을 구현한 클래스. 입력 데이터에 필터(가중치)를 적용하고, 바이어스를 더해
    특성 맵을 생성. 이 계층은 신경망에서 이미지의 공간적 특성을 추출하는 데 사용된다.

    Attributes:
        W (numpy.ndarray): 필터(가중치) 텐서.
        b (numpy.ndarray): 바이어스 벡터.
        stride (int): 필터 적용 시의 이동 간격.
        pad (int): 입력 데이터의 주변을 패딩하는 크기.
    """
    
    def __init__(self, W, b, stride=None, pad=None):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터 저장을 위한 변수
        self.x = None
        self.col = None
        self.col_W = None
        
        # 그래디언트 저장을 위한 변수
        self.dw = None
        self.db = None
        
    
    def forward(self, x):
        """
        순전파 메소드. 입력 데이터에 합성곱 연산을 수행.

        Parameters:
            x (numpy.ndarray): 입력 데이터.

        Returns:
            numpy.ndarray: 합성곱 연산의 결과.
        """
        Filter_Number, Channel, Filter_Height, Filter_Width = self.W.shape
        Number, Channel, Height, Width = x.shape
        out_height = 1 + int((Height + 2 * self.pad -Filter_Height) / self.stride)
        out_width = 1 + int((Width + 2 * self.pad -Filter_Width) / self.stride)
        
        # 입력 데이터를 2차원으로 변환
        col = im2col(x, Filter_Height, Filter_Width, self.stride, self.pad)
        col_W = self.W.reshape(Filter_Number, -1).T
        
        # 합성곱 연산
        out = np.dot(col, col_W) + self.b
        out = out.reshape(Number, out_height, out_width, -1).transpose(0, 3, 1, 2)
        
        # 중간 데이터 저장
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out
    
    
    def backward(self, dout):
        """
        역전파 메소드. 출력 데이터의 그래디언트로부터 입력 데이터, 가중치 및 바이어스에 대한
        그래디언트를 계산.

        Parameters:
            dout (numpy.ndarray): 출력 데이터의 그래디언트.

        Returns:
            numpy.ndarray: 입력 데이터에 대한 그래디언트.
        """
        Filter_Number, Channel, Filter_Height, Filter_Width = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, Filter_Number)
        
        # 바이어스 그래디언트 계산
        self.db = np.sum(dout, axis=0)
        
        # 가중치 그래디언트 계산
        self.dw = np.dot(self.col.T, dout)
        self.dw = self.dw.transpose(1, 0).reshape(Filter_Number, Channel, Filter_Height, Filter_Width)
        
        # 입력 데이터 그래디언트 계산
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, Filter_Height, Filter_Width, self.stride, self.pad)
        
        return dx
    

class Dropout:
    """
    드롭아웃(Dropout)을 구현한 클래스.

    드롭아웃은 네트워크의 과적합을 방지하기 위해 학습 과정에서 
    무작위로 일부 뉴런을 비활성화시키는 기법이다.

    Attributes:
        dropout_ratio (float): 뉴런을 비활성화할 확률.
        mask (numpy.array): 뉴런의 활성화 여부를 결정하는 마스크.
    """

    def __init__(self, dropout_ratio=0.5):
        """
        클래스 초기화 메소드.

        Parameters:
            dropout_ratio (float, optional): 뉴런을 비활성화할 확률. 기본값은 0.5.
        """
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flg=True):
        """
        드롭아웃의 순전파 메소드.

        학습 시에는 무작위로 일부 뉴런을 비활성화하고,
        테스트 시에는 모든 뉴런을 활성화한다.

        Parameters:
            x (numpy.array): 입력 데이터.
            train_flg (bool, optional): 학습 모드 여부. 기본값은 True.

        Returns:
            numpy.array: 드롭아웃이 적용된 출력 데이터.
        """
        if train_flg:
            # 학습 시, 무작위로 뉴런 비활성화
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio  # 생성된 난수 배열의 각 요소가 ratio보다 큰지 작은지 판단하여 Boolean으로 이루어진 배열 생성
            return x * self.mask  # Boolean 배열과 입력 데이터를 곱하여 무작위로 뉴런 비활성화
        else:
            # 테스트 시, 모든 뉴런 활성화 (출력 조정)
            return x * (1.0 - self.dropout_ratio)  # 학습 단계에서 드롭아웃으로 인해 감소된 출력 강도를 보상
        
    def backward(self, dout):
        """
        드롭아웃의 역전파 메소드.

        순전파 때 비활성화된 뉴런에 대한 그래디언트는 0이 된다.

        Parameters:
            dout (numpy.array): 출력에 대한 그래디언트.

        Returns:
            numpy.array: 입력에 대한 그래디언트.
        """
        return dout * self.mask
    

class FC_Layer:
    """
    완전 연결 계층(Fully Connected Layer)을 구현한 클래스.

    이 계층은 입력 데이터를 가중치와 행렬 곱셈을 통해 변환한 후,
    편향(bias)를 추가하여 출력.

    Attributes:
        W (numpy.array): 가중치 행렬.
        b (numpy.array): 바이어스 벡터.
        x (numpy.array): 계층에 입력된 데이터.
        original_x_shape (tuple): 입력 데이터의 원래 형태.
        dw (numpy.array): 가중치에 대한 그래디언트.
        db (numpy.array): 바이어스에 대한 그래디언트.
    """

    def __init__(self, W, b):
        """
        클래스 초기화 메소드.

        Parameters:
            W (numpy.array): 가중치 행렬.
            b (numpy.array): 바이어스 벡터.
        """
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.conv_output_shape = None
        self.dw = None
        self.db = None
        
    def forward(self, x):
        """
        순전파 메소드.

        입력 데이터 x가 완전 연결 계층을 통과.

        Parameters:
            x (numpy.array): 입력 데이터.
            x.shape[0] : 배치 크기

        Returns:
            numpy.array: 계층을 통과한 후의 출력 데이터.
        """
        # 입력 데이터의 원래 형태 저장
        self.original_x_shape = x.shape
        
        # 컨볼루션 계층 또는 풀링 계층의 출력 형태를 저장
        if x.ndim == 4:  # 입력 x가 4차원인 경우(conv or pool)
            self.conv_output_shape = x.shape  # 형태 저장  
        
        # 데이터를 2차원으로 변환
        x = x.reshape(x.shape[0], -1)  # 이미지 데이터라면, 일렬로 펴진 벡터로 변환시키기 위한 과정
        self.x = x

        # 완전 연결 계층의 연산 수행
        out = np.dot(self.x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        """
        역전파 메소드.

        순전파의 출력에 대한 그래디언트를 이용해
        입력 데이터와 매개변수의 그래디언트를 계산한다.

        Parameters:
            dout (numpy.array): 출력에 대한 그래디언트.
            dout의 각 열은 해당 계층의 각 뉴런에 대응한다.

        Returns:
            numpy.array: 입력에 대한 그래디언트.
            역전파 과정에서는 순전파의 연산을 역으로 따라가야 한다. 순전파에서 x의 형태가 변형되었으므로 다시 되돌려야 함.
            dx가 네트워크의 이전 계층으로 올바르게 전달 되어야 하기 때문.
        """
        # 입력에 대한 그래디언트 계산
        # print(f"FC layer의 맨처음 들어오는 dout shape : {dout.shape}")
        dx = np.dot(dout, self.W.T)  # 순전파의 np.dot(self.x, self.W) 의 역연산
        # print(f"FC layer의 np.dot역연산 결과 shape : {dx.shape}")
        
        # 가중치에 대한 그래디언트 계산
        self.dw = np.dot(self.x.T, dout)  # 순전파의 W 가 x와 곱해진 것에 대한 역연산
        # print(f"FC layer의 W와 x 곱의 역연산 shape : {self.dw.shape}")
        
        # 편향(바이어스)의 그래디언트 계산
        self.db = np.sum(dout, axis=0)  # dout의 각 열에 대한 합, 편향은 각 출력 뉴런에 더해지므로, 그래디언트는 dout의 합
        # print(f"FC layer의 편향 shape : {self.db.shape}")

        # 입력 데이터의 그래디언트를 원래 형태로 변환
        # * 연산자를 이용하여 튜플이나 리스트와 같은 반복 가능한(iterable) 객체의 요소를 개별적인 인자로 풀어서(unpack) 전달
        # print(f"FC layer의 *연산 전 dx shape : {dx.shape}")
        dx = dx.reshape(*self.original_x_shape)
        # print(f"FC layer의 *연산 후 dx shape : {dx.shape}")
        
        # print(f"original_x_shape : {self.original_x_shape}")
        
        # 컨볼루션 계층 이전의 형태로 dx 재구성
        if self.conv_output_shape is not None:
            dx = dx.reshape(*self.conv_output_shape)
        
        # print(f"FC layer의 original_shape으로 reshape한 dx shape : {dx.shape}")
        
        return dx
    

class Pooling:
    """
    맥스 풀링(Max pooling) 계층을 구현한 클래스.

    이 계층은 입력 데이터에 대해 맥스 풀링 연산을 수행한다. 
    풀링은 일반적으로 컨볼루션 신경망에서 공간적 차원을 축소하고 데이터의 중요한 특징을 유지하는 데 사용된다.

    Attributes:
        pool_height (int): 풀링 윈도우의 높이.
        pool_width (int): 풀링 윈도우의 너비.
        stride (int): 풀링 윈도우의 스트라이드(이동 간격).
        pad (int): 패딩 크기.
    """
    
    def __init__(self, pool_height, pool_width, stride, pad=0):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.pad = pad
        self.x = None  # 입력 데이터
        self.arg_max = None  # 풀링 시 최댓값의 인덱스
        
    
    def forward(self, x):
        """
        순전파 함수.

        Parameters:
            x (numpy.ndarray): 입력 데이터.

        Returns:
            numpy.ndarray: 풀링 연산 결과.
        """
        
        Number, Channel, Height, Width = x.shape
        self.original_x_shape = x.shape
        
        # print(f"x.shape : {x.shape}")
        out_height = int(1 + (Height - self.pool_height) / self.stride)
        # print(f"out_height : {out_height}")
        out_width = int(1 + (Width - self.pool_width) / self.stride)
        # print(f"out_width : {out_width}")
        
        # 입력 데이터를 2차원 배열로 변환
        col = im2col(x, self.pool_height, self.pool_width, self.stride, self.pad)
        col = col.reshape(-1, self.pool_height * self.pool_width)
        
        arg_max = np.argmax(col, axis=1)  # 최댓값의 위치(인덱스) 추출
        out = np.max(col, axis=1)  # 최댓값 계산
        out = out.reshape(Number, out_height, out_width, Channel).transpose(0, 3, 1, 2)  # 결과 형태 재조정 및 차원 변경
        # print(f"out.shape : {out.shape}")
        
        self.x = x
        # print(f"self.x.shape {self.x.shape}")
        self.arg_max = arg_max
        
        return out
    
    def backward(self, dout):
        """
        역전파 함수.

        Parameters:
            dout (numpy.ndarray): 상위 계층으로부터 전달된 그래디언트.

        Returns:
            numpy.ndarray: 입력 데이터에 대한 그래디언트.
        """
        # dout를 적절한 형태로 전치시킨다
        # 이는 dout의 차원을 풀링 연산에 맞게 재배열하기 위함 -> (Number, Height, Width, Channel)
        # print(f"dout shape : {dout.shape}")
        dout = dout.reshape(self.original_x_shape[0], self.original_x_shape[1], 
                            int((self.original_x_shape[2] - self.pool_height) / self.stride + 1), 
                            int((self.original_x_shape[3] - self.pool_width) / self.stride + 1))
        # print(f"reshaped dout : {dout.shape}")
        dout = dout.transpose(0, 2, 3, 1)
        # print(dout.shape)
        
        # 풀링 영역의 크기를 계산
        pool_size = self.pool_height * self.pool_width
        # dmax를 0으로 초기화합니다. -> 각 풀링 영역에 대한 그래디언트를 저장할 배열
        dmax = np.zeros((dout.size, pool_size))

        # dout의 그래디언트를 풀링 영역에서 최댓값을 가진 위치에 할당한다
        # self.arg_max는 순전파 때 풀링 영역에서 최댓값의 위치를 저장한 배열.
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        # dmax의 형태를 다시 변경하여 col2im 함수에 전달할 수 있는 형태로 만든다.
        dcol = dmax.reshape(dout.shape[0], dout.shape[1] * dout.shape[2], -1)
        # col2im 함수를 사용하여 dcol을 원래 이미지 데이터 형태의 그래디언트(dx)로 변환한다.
        dx = col2im(dcol, self.x.shape, self.pool_height, self.pool_width, self.stride, self.pad)

        return dx


def softmax(x):
    """
    소프트맥스(softmax) 함수를 구현한 함수.
    입력된 데이터에 대해 소프트맥스 변환을 수행하여, 결과를 확률 분포 형태로 반환.

    Parameters:
        x (numpy.ndarray): 입력 데이터. 1차원 또는 2차원 배열일 수 있다.
        1차원 : 단일 이미지에 대한 클래스 분류 or 하나의 문장에 대한 감정 분석 등... -> 개별 샘플
        2차원 : 여러 이미지에 대한 분류 또는 여러 문장에 대한 처리 -> 배치 처리된 샘플

    Returns:
        numpy.ndarray: 소프트맥스 변환 결과. 입력과 동일한 형태를 가집니다.
    """
    if x.ndim == 2:
        # 2차원 입력 데이터 처리
        x = x.T  # 전치
        x = x - np.max(x, axis=0)  # 오버플로 방지를 위해 최대값을 빼줌
        y = np.exp(x) / np.sum(np.exp(x), axis=0)  # 소프트맥스 계산
        return y.T  # 결과를 원래 형태로 전치하여 반환

    # 1차원 입력 데이터 처리
    x = x - np.max(x)  # 오버플로 방지
    return np.exp(x) / np.sum(np.exp(x))  # 소프트맥스 계산


def cross_entropy_error(y, t):
    """
    크로스 엔트로피 손실을 계산하는 함수.

    Parameters:
        y (numpy.ndarray): 신경망의 출력 (예측 확률 분포).
        t (numpy.ndarray): 실제 레이블 (원-핫 벡터 또는 레이블 인덱스).

    Returns:
        float: 계산된 크로스 엔트로피 손실 값.
    """
    # y가 1차원 배열일 경우, 2차원 배열로 변환
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # 실제 레이블을 2차원 배열로 변환
        y = y.reshape(1, y.size)  # 예측 확률 분포를 2차원 배열로 변환
    
    # t가 원-핫 벡터일 경우, 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)  # 각 샘플의 실제 클래스 인덱스를 추출
    
    batch_size = y.shape[0]  # 배치 크기 계산

    # 크로스 엔트로피 손실 계산
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class SoftmaxWithLoss:
    """
    소프트맥스 함수와 크로스 엔트로피 손실을 결합한 계층.

    이 계층은 신경망의 마지막에 위치하며, 분류 문제에서 활용됩니다.
    소프트맥스 함수를 통해 예측 값을 확률 분포로 변환하고, 크로스 엔트로피 손실을 통해
    예측 값과 실제 레이블 간의 차이를 계산합니다.
    """

    def __init__(self):
        self.loss = None  # 손실
        self.y = None     # 소프트맥스 출력
        self.t = None     # 실제 레이블

    def forward(self, x, t):
        """
        순전파 함수.

        Parameters:
            x (numpy.ndarray): 입력 데이터.
            t (numpy.ndarray): 실제 레이블.

        Returns:
            float: 계산된 손실 값.
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """
        역전파 함수.

        Parameters:
            dout (float): 상위 계층에서 전달된 미분값.

        Returns:
            numpy.ndarray: 입력 데이터에 대한 그래디언트.
        """
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:  # 원-핫 인코딩 형태의 레이블
            dx = (self.y - self.t) / batch_size
        else:  # 레이블 인덱스 형태의 레이블
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
    

class Adam:
    """
    Adam(Adaptive Moment Estimation) 최적화 알고리즘을 구현한 클래스.

    Adam은 모멘텀과 RMSProp의 아이디어를 결합한 알고리즘으로,
    각 매개변수에 대해 적응적인 학습률을 조정한다.

    Attributes:
        lr (float): 학습률.
        beta1 (float): 모멘텀을 위한 감쇠율.
        beta2 (float): 스케일링된 제곱 그래디언트를 위한 감쇠율.
        iter (int): 업데이트 횟수.
        m (dict): 모멘텀.
        v (dict): RMSProp.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """
        클래스 초기화 메소드.

        Parameters:
            lr (float, optional): 학습률. 기본값은 0.001.
            beta1 (float, optional): 모멘텀을 위한 감쇠율. 기본값은 0.9.
            beta2 (float, optional): 스케일링된 제곱 그래디언트를 위한 감쇠율. 기본값은 0.999.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        """
        매개변수를 업데이트하는 메소드.

        Parameters:
            params (dict): 최적화할 매개변수들을 포함하는 딕셔너리.
            grads (dict): 각 매개변수의 그래디언트를 포함하는 딕셔너리.
        """
        if self.m is None:
            # 최초 호출 시, 모멘텀과 RMSProp 변수 초기화
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for key in params.keys():
            # 모멘텀과 RMSProp 업데이트
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key]**2)
            
            # 매개변수 업데이트
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            

class EarlyStopping:
    def __init__(self, patience=None, verbose=False, delta=0, base_path='Checkpoint', save_path='./'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.base_path = base_path
        self.save_path = save_path
    
    
    def save_checkpoint(self, val_loss, model, epoch):
        checkpoint_path = os.path.join(self.save_path, f"{self.base_path}_Best_{epoch+1}.npz")
        np.savez(checkpoint_path, **model.params)
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {checkpoint_path} ...")
        self.val_loss_min = val_loss
        
    def step(self, val_loss, model, epoch):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping Count : {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0
            

def get_grad(f, x):
    """
    주어진 함수 f에 대한 x의 위치에서의 기울기(그래디언트)를 구하기.

    중앙 차분(central difference) 방식을 사용하여 미분을 표현.
    (f(x+h) - f(x-h)) / 2h 공식을 사용.

    Parameters:
        f (callable): 미분할 함수.
        x (numpy.array): 기울기를 계산할 위치의 배열.

    Returns:
        numpy.array: 함수 f의 x에서의 기울기.
    """

    h = 1e-4  # 미분을 위한 작은 변화량, 0.0001
    grad = np.zeros_like(x)  # x와 같은 형태의 배열을 0으로 초기화

    # numpy.nditer를 사용하여 배열의 각 요소에 접근
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index  # 현재 위치의 인덱스

        tmp_val = x[idx]  # 원래 값 임시 저장

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 중앙 차분을 이용한 기울기 계산
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 원래 값으로 복원
        x[idx] = tmp_val
        it.iternext()
        
    return grad


def graph(data_list, title, color, save_path):
    batch_num_list = [i for i in range(0, len(data_list))]
    
    plt.figure(figsize=(20, 10))
    plt.rc('font', size=25)
    plt.plot(batch_num_list, data_list, color=color, marker='o', linestyle='solid')
    plt.title(title)
    plt.xlabel('Epoch')
    
    title = plt.ylabel(title)

    plt.savefig(save_path, dpi=600)
    # plt.show()
    plt.close()
    

def split_dataset(src_folder, dst_base_folder, ratios=[0.8, 0.1, 0.1]):
    """
    지정된 소스 폴더(src_folder)에서 파일을 읽어와서, 
    훈련(train), 검증(val), 테스트(test) 폴더로 분할하는 함수.

    이 함수는 소스 폴더의 하위 폴더 구조를 유지하며, 지정된 비율에 따라 파일을 무작위로 분할한다.

    Parameters:
        src_folder (str): 소스 폴더의 경로.
        dst_base_folder (str): 대상 폴더의 기본 경로. 이 폴더 내에 train, val, test 폴더가 생성됨.
        ratios (list): 훈련, 검증, 테스트 데이터셋으로 분할할 비율의 리스트. 기본값은 [0.8, 0.1, 0.1].
    """
    
    # 대상 폴더의 경로 생성
    dst_folders = [os.path.join(dst_base_folder, name) for name in ["train", "val", "test"]]

    # 대상 폴더가 존재하지 않는 경우 생성
    for folder in dst_folders:
        os.makedirs(folder, exist_ok=True)

    # 소스 폴더의 모든 하위 폴더 검색
    sub_folders = [f.path for f in os.scandir(src_folder) if f.is_dir()]

    # 각 하위 폴더의 이미지를 처리
    for sub in sub_folders:
        sub_folder_name = os.path.basename(sub)  # 하위 폴더 이름 추출
        images = [img for img in os.listdir(sub) if img.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)  # 이미지 리스트를 무작위로 섞음

        # 비율 리스트 복제 (원본 수정 방지)
        temp_ratios = ratios.copy()

        # 각 이미지를 무작위로 선택된 폴더에 복사
        for img_file in images:
            selected_folder = random.choices(dst_folders, weights=temp_ratios, k=1)[0]

            # 선택된 폴더의 비율 조정
            idx = dst_folders.index(selected_folder)
            temp_ratios[idx] -= 1 / len(images)

            # 최종 대상 폴더에 동일한 하위 폴더 구조 생성
            final_destination = os.path.join(selected_folder, sub_folder_name)
            os.makedirs(final_destination, exist_ok=True)

            # 이미지 파일을 최종 대상 폴더로 복사
            shutil.copy(os.path.join(sub, img_file), os.path.join(final_destination, img_file))
            

def visualize_result(model, data, labels, num_samples=5):
    """
    분류 모델의 결과를 시각화하는 함수.

    이 함수는 주어진 데이터의 샘플을 모델에 입력하고, 예측 결과를 시각적으로 표시한다.

    Parameters:
        model: 분류 모델.
        data (torch.Tensor): 입력 데이터.
        labels (torch.Tensor): 실제 레이블.
        num_samples (int): 시각화할 샘플의 수. 기본값은 5.
    """
    
    # 넘파이 기반 모델 예측 수행
    outputs = model.predict(data)
    predicted = np.argmax(outputs, axis=1)

    if num_samples is None or num_samples > len(data):
        num_samples = len(data)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        # 원본 이미지
        img = np.transpose(data[i], (1, 2, 0)) # 데이터 형태가 (C, H, W)인 경우
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original - Label: {}'.format(labels[i]))
        axes[i, 0].axis('off')

        # 예측 레이블
        axes[i, 1].text(0.5, 0.5, 'Predicted: {}'.format(predicted[i]), 
                        fontsize=14, ha='center')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

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
            self.params['W' + str(idx + 1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        
        # BatchNormalization 파라미터 추가
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4]):
            self.params['gamma' + str(idx + 1)] = np.ones(conv_param['filter_num'])
            self.params['beta' + str(idx + 1)] = np.zeros(conv_param['filter_num'])

        # 완전 연결 계층의 가중치와 편향 초기화
        self.params['W5'] = weight_init_scales[4] * np.random.randn(64 * 7 * 7, hidden_size)
        self.params['b5'] = np.zeros(hidden_size)
        self.params['W6'] = weight_init_scales[5] * np.random.randn(hidden_size, output_size)
        self.params['b6'] = np.zeros(output_size)

        # ======= 계층 생성 =======
        self.layers = []

        # 첫 번째 컨볼루션 계층
        # 입력 형태: (N, C, H, W) -> 여기서 N은 배치 크기, C는 채널 수, H와 W는 이미지의 높이와 너비
        # (32, 3, 224, 224)
        # 출력 형태: (N, F1, H1, W1) -> 여기서 F1은 첫 번째 컨볼루션 계층의 필터 개수
        # (32, 16, 112, 112)
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(BatchNormalization(self.params['gamma1'], self.params['beta1']))
        self.layers.append(ReLU())

        # 두 번째 컨볼루션 계층
        # 입력 형태: (N, F1, H1, W1)
        # (32, 16, 112, 112)
        # 출력 형태: (N, F2, H2, W2) -> 여기서 F2는 두 번째 컨볼루션 계층의 필터 개수
        # (32, 32, 112, 112) -> F2=32, H2=112, W2=112
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(BatchNormalization(self.params['gamma2'], self.params['beta2']))
        self.layers.append(ReLU())
        
        # 첫 번째 풀링 계층
        # 입력 형태: (32, 32, 112, 112) -> 두 번째 컨볼루션 계층의 출력
        # 출력 형태: (32, 32, 56, 56) -> H2'=56, W2'=56
        self.layers.append(Pooling(pool_height=2, pool_width=2, stride=2))

        # 세 번째 컨볼루션 계층
        # 입력 형태: (N, F2, H2, W2)
        # (32, 32, 56, 56)
        # 출력 형태: (N, F3, H3, W3) -> 여기서 F3는 세 번째 컨볼루션 계층의 필터 개수
        # (32, 32, 56, 56)
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(BatchNormalization(self.params['gamma3'], self.params['beta3']))
        self.layers.append(ReLU())

        # 네 번째 컨볼루션 계층
        # 입력 형태: (N, F3, H3, W3)
        # (32, 32, 56, 56)
        # 출력 형태: (N, F4, H4, W4) -> 여기서 F4는 네 번째 컨볼루션 계층의 필터 개수
        # (32, 64, 28, 28) -> F4=64, H4=28, W4=28
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(BatchNormalization(self.params['gamma4'], self.params['beta4']))
        self.layers.append(ReLU())
        
        # 두 번째 풀링 계층
        # 입력 형태: (32, 64, 28, 28) -> 네 번째 컨볼루션 계층의 출력
        # 출력 형태: (32, 64, 14, 14) -> H4'=14, W4'=14
        self.layers.append(Pooling(pool_height=2, pool_width=2, stride=2))

        # 첫 번째 완전 연결 계층
        # 입력 형태: (32, 64*14*14) -> 풀링 계층의 출력을 평탄화
        # 출력 형태: (32, 50) -> hidden_size=50
        self.layers.append(FC_Layer(self.params['W5'], self.params['b5']))
        self.layers.append(ReLU())
        self.layers.append(Dropout(0.5))
        
        # 두 번째 완전 연결 계층
        # 입력 형태: (32, 50) -> 첫 번째 완전 연결 계층의 출력
        # 출력 형태: (32, 2) -> output_size=2 (분류 클래스 수)
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
            if isinstance(layer, BatchNormalization) or isinstance(layer, Pooling):
                x = layer.forward(x)
            elif isinstance(layer, FC_Layer):
                # 컨볼루션 계층 출력을 평탄화
                x = x.reshape(x.shape[0], -1)
                x = layer.forward(x)
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
                grads['W' + str(i+1)] = layer.dw
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
                layer.w = self.params['W' + str(i+1)]
                layer.b = self.params['b' + str(i+1)]
            if isinstance(layer, BatchNormalization):
                layer.gamma = self.params['gamma' + str(i+1)]
                layer.beta = self.params['beta' + str(i+1)]

        print("Params loaded successfully:", self.params.keys())
        

class Trainer:
    """
    신경망 모델의 훈련을 관리하는 클래스.

    이 클래스는 신경망 모델의 훈련을 위한 여러 기능을 제공한다. 
    훈련 데이터와 테스트 데이터에 대한 로딩, 모델의 업데이트, 
    훈련과 검증 과정의 정확도 계산 등을 수행합니다.

    Parameters:
        network: 훈련할 신경망 모델.
        x_train_loader: 훈련 데이터 로더.
        x_test_loader: 테스트 데이터 로더.
        epochs (int): 에폭 수.
        mini_batch_size (int): 미니 배치 크기.
        optimizer (str): 최적화 알고리즘의 이름.
        optimizer_param (dict): 최적화 알고리즘의 매개변수.
        evaluate_sample_num_per_epoch (int): 에폭당 평가할 샘플 수.
        verbose (bool): 상세 정보 출력 여부.
    """

    def __init__(self, network, x_train_loader, x_test_loader,
                 epochs=30, mini_batch_size=32,
                 optimizer='adam', optimizer_param={'lr':0.0001}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        
        # 네트워크와 데이터 로더 초기화
        self.network = network
        self.train_loader = x_train_loader
        self.test_loader = x_test_loader
        
        # 훈련 설정 초기화
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.verbose = verbose

        # 최적화 알고리즘 초기화
        optimizer_class_dict = {'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        # 훈련 데이터 크기 및 에폭당 반복 횟수 설정
        self.train_size = len(x_train_loader.dataset) # x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.current_iter = 0
        self.current_epoch = 0
        
        # 훈련 및 검증 과정의 기록을 저장하는 리스트
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        
        # 현재 학습중인지에 대한 상태
        self.train_mode = True


    def train_step(self):
        """
        하나의 에폭 동안 모델을 훈련하는 단계.
        """
        
        # 현재 훈련 상태에 따라 적절한 데이터 로더 선택
        dataloader = self.train_loader if self.train_mode else self.test_loader
        name = "train" if self.train_mode else "evaluate"

        # tqdm을 이용하여 진행 상황 시각화
        for x_batch, t_batch in tqdm(dataloader, desc=name):
            # 그래디언트 계산 및 최적화
            grads = self.network.gradient(x_batch, t_batch)
            
            # network.params에 없는 키를 grads에서 제거
            keys_to_remove = [key for key in grads.keys() if key not in self.network.params]
            for key in keys_to_remove:
                del grads[key]

            self.optimizer.update(self.network.params, grads)
            
            # 손실 계산
            loss = self.network.loss(x_batch, t_batch)
            
            # 훈련 중 손실 기록 및 wandb 로깅
            if self.train_mode:
                self.train_loss_list.append(loss)
                wandb.log({"train_loss": loss})
            
            # 상세 정보 출력    
            if self.verbose: 
                print(f"\t{name} loss: {loss}")

            # 에폭의 마지막에서 정확도 계산
            if self.current_iter % self.iter_per_epoch == 0:
                self.current_epoch += 1

                if self.train_mode:
                    train_acc = self.network.accuracy(x_batch, t_batch)
                    self.train_acc_list.append(train_acc)
                    wandb.log({"train_accuracy": train_acc})
                else:
                    test_acc = self.calculate_accuracy(self.test_loader)
                    self.test_acc_list.append(test_acc)
                    wandb.log({"test_accuracy": test_acc})

                    if self.verbose: 
                        print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
            self.current_iter += 1


    def train(self, current_epochs):
        """
        신경망 모델을 훈련하는 메소드.

        Args:
            current_epochs (int): 현재까지 완료된 에폭 수.
        """
        for epoch in range(self.epochs):
            self.train_step()
            
            if (epoch + 1) % 5 ==  0:
                test_data, test_labels = next(iter(self.test_loader))
                visualize_result.visualize_result(self.network, test_data, test_labels)
            
            # 모델 파라미터 저장 및 그래프 그리기
            self.network.save_params(file_name=f"epoch_{epoch+current_epochs+1}.pkl")
            print(f"model({epoch+1}/{self.epochs}) is saved!")
            graph(self.train_loss_list, 'loss', 'red', f"epoch_{epoch+current_epochs+1}")


    def test(self):
        """
        신경망 모델의 테스트 데이터에 대한 정확도를 평가하는 메소드.
        """
        hits = 0.0
        # 테스트 데이터 로더를 통해 정확도 계산
        for imgs, labels in self.test_loader:
            hits += self.network.accuracy(imgs, labels)
        test_acc = hits / len(self.test_loader.dataset)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


    def calculate_accuracy(self, loader):
        """
        데이터 로더를 통해 데이터셋의 정확도를 계산하는 메소드.

        Args:
            loader: 데이터 로더.

        Returns:
            float: 계산된 정확도.
        """
        hits = 0.0
        for imgs, labels in loader:
            hits += self.network.accuracy(imgs, labels)
        return hits / len(loader.dataset)
    
    
#############################################################################################

device = "cuda" if torch.cuda.is_available() else "CPU"

# 난수 생성을 위한 시드 설정
SEED = 777

# CPU 환경을 위한 PyTorch 난수 생성기에 시드 설정
torch.manual_seed(SEED)

# 현재 GPU를 위한 PyTorch 난수 생성기에 시드 설정
torch.cuda.manual_seed(SEED)

# cuDNN의 결정적(deterministic) 알고리즘 사용 여부 설정
# False로 설정할 경우, 비결정적 알고리즘이 허용되어 성능이 향상될 수 있지만, 재현성이 감소할 수 있음
torch.backends.cudnn.deterministic = False

# cuDNN에서 최적의 알고리즘을 자동으로 찾도록 설정
# True로 설정할 경우, 고정된 입력 크기에 대해 더 빠른 성능을 제공하지만, 다양한 크기의 입력에서는 성능 저하가 발생할 수 있음
torch.backends.cudnn.benchmark = True

# 데이터셋 디렉토리 설정 해야함
data_dir = r'/mnt/hdd_2T/ivan/data/'

# albumentations을 사용한 데이터 변환 정의
data_transforms = {
    'train': A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=.1),
            A.Blur(blur_limit=3, p=.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Resize(224, 224),
        ToTensorV2()
    ]),
    'val': A.Compose([
        A.Resize(224, 224),
        ToTensorV2()
    ])
}

# 데이터셋 로딩
split_dir = r'/mnt/hdd_2T/ivan/data/split_data'

class MaskDataset(Dataset):
    """
    마스크 착용 여부를 구별하는 이미지 데이터셋 클래스.

    이 클래스는 주어진 디렉토리에서 마스크 착용 여부를 구분하는 이미지 데이터셋을 로드하고,
    albumentations 라이브러리를 사용한 이미지 변환을 적용한다.

    Parameters:
        data_dir (str): 데이터셋이 위치한 디렉토리 경로.
        mode (str): 데이터셋 모드 ('train', 'val', 'test').
        transform (albumentations.Compose): 이미지에 적용할 변환.
    """
    def __init__(self, data_dir, mode, transform=None):
        self.all_data = sorted(glob.glob(os.path.join(data_dir, mode, '*', '*')))
        self.transform = transform
        
    
    def __getitem__(self, index):
        """
        인덱스에 해당하는 데이터를 반환합니다.

        Parameters:
            index (int): 데이터셋에서 가져올 샘플의 인덱스.

        Returns:
            tuple: (변환된 이미지, 레이블).
        """
        data_path = self.all_data[index]
        img = Image.open(data_path)
        label = 0 if os.path.basename(data_path).startswith("mask") else 1
        
        if self.transform:
            img = self.transform(image=np.array(img))["image"]
            
        return img, label
    
    def __len__(self):
        """
        데이터셋의 전체 길이를 반환합니다.

        Returns:
            int: 데이터셋의 전체 길이.
        """
        return len(self.all_data)

# 데이터셋 분할(필요할 시, 주석 해체 후 사용)
# split_dataset(data_dir, split_dir)

train_dataset = MaskDataset(split_dir, 'train', transform=data_transforms['train'])
val_dataset = MaskDataset(split_dir, 'val', transform=data_transforms['val'])
test_dataset = MaskDataset(split_dir, 'test', transform=data_transforms['val'])

batch_size = 32
num_epochs = 300
learning_rate = 0.0001
num_workers = 0

# 데이터로더
# train_loader = DataLoader(shuffled_trainset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=num_workers)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)

model = VGG6(
    input_dim=(3, 224, 224),
    conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 2},
    conv_param_2={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
    conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 2},
    conv_param_4={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 2},
    hidden_size=50,
    output_size=2  
    )

trainer = Trainer(model, x_train_loader = train_loader, x_test_loader = test_loader,
                          epochs=num_epochs, mini_batch_size=batch_size,
                          optimizer='adam', evaluate_sample_num_per_epoch=5)

# 현재 시간에서 분까지만 잘라내기
now = time.localtime()
# YYYYMMDDHHMM 형태로 저장
now = time.strftime('%Y%m%d%H%M', now)

checkout_path = fr"./2-2_refactoring/checkout/{now}"
os.makedirs(checkout_path, exist_ok=True)

wandb.init(project='Mask_Classification', name='numpy_without_custom_data_shuffle_VGG6',
           config={
               'learning_rate' : 0.001,
               'epochs' : 300,
               'batch_size' : batch_size,
               'dataset' : 'kaggle',
               'architecture' : 'VGG6',
               'optimizer' : 'Adam',
               'criterion' : 'Cross Entropy Loss',
               'lr_scheduler' : 'None',
               'amp' : None,
               'pin_memory' : True,
               'non_blocking' : None,
               'accumulation_steps' : None,
               'num_workers' : num_workers,
               'EarlyStopping' : True
})

config = wandb.config

# Early Stop 설정
early_stop = EarlyStopping(patience=10, verbose=True, save_path=checkout_path)

# 학습 시작
train_losses, val_losses = trainer.train(current_epochs=0)

# W&b 종료
wandb.finish()
