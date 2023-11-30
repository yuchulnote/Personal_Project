import numpy as np

def softmax(x):
    """
    소프트맥스(softmax) 함수를 구현한 함수.
    입력된 데이터에 대해 소프트맥스 변환을 수행하여, 결과를 확률 분포 형태로 반환.

    Parameters:
        x (numpy.ndarray): 입력 데이터. 1차원 또는 2차원 배열일 수 있다.
        1차원 : 단일 이미지에 대한 클래스 분류 or 하나의 문장에 대한 감정 분석 등... -> 개별 샘플
        2차원 : 여러 이미지에 대한 분류 또는 여러 문장에 대한 처리 -> 배치 처리된 샘플

    Returns:
        numpy.ndarray: 소프트맥스 변환 결과. 입력과 동일한 형태를 가집니다.
    """
    if x.ndim == 2:
        # 2차원 입력 데이터 처리
        x = x.T  # 전치
        x = x - np.max(x, axis=0)  # 오버플로 방지를 위해 최대값을 빼줌
        y = np.exp(x) / np.sum(np.exp(x), axis=0)  # 소프트맥스 계산
        return y.T  # 결과를 원래 형태로 전치하여 반환

    # 1차원 입력 데이터 처리
    x = x - np.max(x)  # 오버플로 방지
    return np.exp(x) / np.sum(np.exp(x))  # 소프트맥스 계산
