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
            self.conv_output_shape = x.shpae  # 형태 저장
        
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
        dx = np.dot(dout, self.W.T)  # 순전파의 np.dot(self.x, self.W) 의 역연산
        
        # 가중치에 대한 그래디언트 계산
        self.dw = np.dot(self.x.T, dout)  # 순전파의 W 가 x와 곱해진 것에 대한 역연산
        
        # 편향(바이어스)의 그래디언트 계산
        self.db = np.sum(dout, axis=0)  # dout의 각 열에 대한 합, 편향은 각 출력 뉴런에 더해지므로, 그래디언트는 dout의 합

        # 입력 데이터의 그래디언트를 원래 형태로 변환
        # * 연산자를 이용하여 튜플이나 리스트와 같은 반복 가능한(iterable) 객체의 요소를 개별적인 인자로 풀어서(unpack) 전달
        dx = dx.reshape(*self.original_x_shape)
        
        # 컨볼루션 계층 이전의 형태로 dx 재구성
        if self.conv_output_shape is not None:
            dx = dx.reshape(*self.conv_output_shape)
        
        return dx
