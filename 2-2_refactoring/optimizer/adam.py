import numpy as np

class Adam:
    """
    Adam(Adaptive Moment Estimation) 최적화 알고리즘을 구현한 클래스.

    Adam은 모멘텀과 RMSProp의 아이디어를 결합한 알고리즘으로,
    각 매개변수에 대해 적응적인 학습률을 조정합니다.

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
