import numpy as np

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
        self.dw = None
        self.db = None
        
    def forward(self, x):
        """
        순전파 메소드.

        입력 데이터 x가 완전 연결 계층을 통과.

        Parameters:
            x (numpy.array): 입력 데이터.

        Returns:
            numpy.array: 계층을 통과한 후의 출력 데이터.
        """
        # 입력 데이터의 원래 형태 저장
        self.original_x_shape = x.shape
        # 데이터를 2차원으로 변환
        x = x.reshape(x.shape[0], -1)
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

        Returns:
            numpy.array: 입력에 대한 그래디언트.
        """
        # 입력 데이터와 가중치에 대한 그래디언트 계산
        dx = np.dot(dout, self.W.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # 입력 데이터의 그래디언트를 원래 형태로 변환
        dx = dx.reshape(*self.original_x_shape)
        
        return dx
