import numpy as np

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
