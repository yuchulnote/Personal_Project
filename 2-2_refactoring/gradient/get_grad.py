import numpy as np

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
