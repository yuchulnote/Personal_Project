import os
import sys

# 현재 스크립트 파일의 절대 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# 현재 디렉토리의 부모 디렉토리 경로 (즉, '2-2_refactoring' 경로)
parent_dir = os.path.dirname(current_dir)

# 시스템 경로에 부모 디렉토리 추가
sys.path.append(parent_dir)

import numpy as np
from commons.im2col import im2col
from commons.col2im import col2im


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
        
        out_height = int(1 + (Height - self.pool_height) / self.stride)
        out_width = int(1 + (Width - self.pool_width) / self.stride)
        
        # 입력 데이터를 2차원 배열로 변환
        col = im2col(x, self.pool_height, self.pool_width, self.stride, self.pad)
        col = col.reshape(-1, self.pool_height * self.pool_width)
        
        arg_max = np.argmax(col, axis=1)  # 최댓값의 위치(인덱스) 추출
        out = np.max(col, axis=1)  # 최댓값 계산
        out = out.reshape(Number, out_height, out_width, Channel).transpose(0, 3, 1, 2)  # 결과 형태 재조정 및 차원 변경
        
        self.x = x
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
        # 역전파에서 dout의 형태를 원래 입력 데이터 x의 형태로 재구성
        dout = dout.reshape(self.original_x_shape[0], self.original_x_shape[1], 
                            int((self.original_x_shape[2] - self.pool_height) / self.stride + 1), 
                            int((self.original_x_shape[3] - self.pool_width) / self.stride + 1))
        
        # dout를 적절한 형태로 전치시킨다
        # 이는 dout의 차원을 풀링 연산에 맞게 재배열하기 위함 -> (Number, Height, Width, Channel)
        dout = dout.transpose(0, 2, 3, 1)

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
