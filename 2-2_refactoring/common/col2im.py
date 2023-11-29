import numpy as np

def col2im(col, input_shape, filter_height, filter_width, stride=1, pad=0):
    """
    2차원 배열을 원래의 이미지 데이터 형태로 변환하는 함수.

    주어진 2차원 배열을 컨볼루션 연산의 입력 형태로 되돌린다.
    이 함수는 컨볼루션 계층의 역전파에서 사용한다.

    Parameters:
        col (numpy.array): 2차원 배열.
        input_shape (tuple): 원래 입력 데이터의 형태 (배치 크기, 채널 수, 높이, 너비).
        filter_height (int): 필터의 높이.
        filter_width (int): 필터의 너비.
        stride (int, optional): 스트라이드 값. 기본값은 1.
        pad (int, optional): 패딩 값. 기본값은 0.

    Returns:
        numpy.array: 변환된 이미지 데이터.
    """
    # 입력 데이터의 형태 추출 및 출력 데이터 형태 계산
    Number, Channel, Height, Width = input_shape
    out_height = (Height + 2 * pad - filter_height) // stride + 1
    out_width = (Width + 2 * pad - filter_width) // stride + 1

    # 2차원 배열을 원래의 4차원 형태로 변환
    # reshape으로 원래의 6차원으로 변경
    # transpose로 col : [Number, Channel, filter_height, filter_width, out_height, out_width] 원래 형태로 변경
    col = col.reshape(Number, out_height, out_width, Channel, filter_height, filter_width).transpose(0, 3, 4, 5, 1, 2)

    # 결과 이미지를 저장할 배열 초기화
    img = np.zeros((Number, Channel, Height + 2 * pad + stride - 1, Width + 2 * pad + stride - 1))

    # 이미지 데이터 재구성
    for y in range(filter_height):
        y_max = y + stride * out_height
        for x in range(filter_width):
            x_max = x + stride * out_width
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 패딩 제거하여 최종 이미지 추출
    return img[:, :, pad:Height + pad, pad:Width + pad]
