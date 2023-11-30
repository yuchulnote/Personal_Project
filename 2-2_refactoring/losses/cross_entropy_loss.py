import numpy as np

def cross_entropy_error(y, t):
    """
    크로스 엔트로피 손실을 계산하는 함수.

    Parameters:
        y (numpy.ndarray): 신경망의 출력 (예측 확률 분포).
        t (numpy.ndarray): 실제 레이블 (원-핫 벡터 또는 레이블 인덱스).

    Returns:
        float: 계산된 크로스 엔트로피 손실 값.
    """
    # y가 1차원 배열일 경우, 2차원 배열로 변환
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # 실제 레이블을 2차원 배열로 변환
        y = y.reshape(1, y.size)  # 예측 확률 분포를 2차원 배열로 변환
    
    # t가 원-핫 벡터일 경우, 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)  # 각 샘플의 실제 클래스 인덱스를 추출
    
    batch_size = y.shape[0]  # 배치 크기 계산

    # 크로스 엔트로피 손실 계산
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
