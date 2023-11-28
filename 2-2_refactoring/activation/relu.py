import copy

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
