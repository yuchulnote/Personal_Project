import numpy as np
import softmax
import cross_entropy_loss

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
