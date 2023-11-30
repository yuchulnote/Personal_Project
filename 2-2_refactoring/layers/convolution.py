import numpy as np
from common.im2col import im2col
from common.col2im import col2im

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
        self.dW = None
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
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(Filter_Number, Channel, Filter_Height, Filter_Width)
        
        # 입력 데이터 그래디언트 계산
        dcol = np.dot(dout, self.col.T, dout)
        dx = col2im(dcol, self.x.shape, Filter_Height, Filter_Width, self.stride, self.pad)
        
        return dx