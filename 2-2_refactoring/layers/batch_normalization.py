import numpy as np

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

        self.running_mean = running_mean if running_mean is not None else np.zeros_like(gamma)
        self.running_var = running_var if running_var is not None else np.zeros_like(gamma)

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
        if x.ndim != 2:
            Number, Channel, Height, Width = x.shape
            x = x.reshape(Number, -1)

        out = self.__forward(x, train_flag)
        return out.reshape(*self.input_shape)

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

        out = self.gamma * normalized_input + self.beta
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
