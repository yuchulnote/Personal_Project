import numpy as np

def im2col(input_data, filter_height, filter_width, stride=1, pad=0):
    """
    다수의 이미지 데이터를 2차원 배열로 변환(평탄화).

    주어진 입력 데이터에 대해 지정된 필터 크기, 스트라이드, 패딩을 적용하여
    컨볼루션 연산을 수행하기 위해 2차원 배열로 변환한다.
    
    컨볼루션 연산을 단순한 두 행렬 곱셈으로 수행하기 위한 함수.

    Parameters:
        input_data (numpy.array): 4차원 입력 데이터 (배치 크기, 채널 수, 높이, 너비).
        filter_height (int): 필터의 높이.
        filter_width (int): 필터의 너비.
        stride (int, optional): 스트라이드 값. 기본값은 1.
        pad (int, optional): 패딩 값. 기본값은 0.

    Returns:
        numpy.array: 변환된 2차원 배열.
    """
    # 입력 데이터의 차원을 추출
    Number, Channel, Height, Width = input_data.shape

    # 출력 데이터의 높이와 너비 계산
    out_height = (Height + 2 * pad - filter_height) // stride + 1
    out_width = (Width + 2 * pad - filter_width) // stride + 1
    """
    입력 데이터에 패딩 적용, 입력 데이터 : (배치크기, 채널 수, 높이, 너비)    
    배치 크기에 대한 패딩은 없으므로 (0, 0)
    채널 수에 대한 패딩은 없으므로 (0, 0)
    높이와 너비에 패딩을 추가, 각 차원의 양쪽에 추가될 패딩의 크기
    'constant' : 상수값으로 패딩을 채운다는 의미, 기본값은 0
    """
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    # 2차원 배열로 변환할 때 사용할 임시 배열 생성
    col = np.zeros((Number, Channel, filter_height, filter_width, out_height, out_width))
    
    # 필터를 적용하여 2차원 배열 생성
    for y in range(filter_height):
        y_max = y + stride * out_height
        for x in range(filter_width):
            x_max = x + stride * out_width
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    """        
    2차원 배열로 변환
    바뀐 col : [Number, out_height, out_width, Channel, filter_height, filter_width]
    reshape된 2차원 col : [Number * out_height * out_width, Channel * filter_height * filter_width]
    axis=0 -> Number * out_height * out_width -> 입력 데이터의 모든 위치에서 적용될 수 있는 필터 영역
    axis=1 -> Channel * filter_height * filter_width -> 필터의 각 영역 내의 원소들을 평탄화한 형태
    """
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(Number * out_height * out_width, -1)
    
    return col
