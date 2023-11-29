import numpy as np

def shuffle_dataset(x, t):
    """
    데이터셋을 무작위로 섞는 함수.

    입력 데이터(x)와 타겟 데이터(t)를 동일한 순서로 섞어서,
    데이터의 순서에 따른 편향을 방지한다.

    Parameters:
        x (numpy.array): 입력 데이터.
        t (numpy.array): 타겟 데이터.

    Returns:
        numpy.array: 섞인 입력 데이터.
        numpy.array: 섞인 타겟 데이터.
    """
    # 데이터의 길이에 따른 무작위 순열 생성
    permutation = np.random.permutation(x.shape[0])  # 0부터 데이터수-1 까지의 숫자를 무작위로 섞은 배열

    # x 데이터가 2차원인 경우와 그 이상인 경우를 구분하여 섞음
    # 2차원인 경우 : (샘플 수, 특성 수), 4차원인 경우 : (샘플 수, 높이, 너비, 채널 수)
    # 샘플 수 부분을 permutation 배열에 따라 재배열하여 섞는다. 
    x_shuffled = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    
    # 타겟 데이터(t) 섞기
    t_shuffled = t[permutation]
    
    return x_shuffled, t_shuffled
