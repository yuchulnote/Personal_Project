from torch.utils.data.dataset import random_split


def split_dataset(dataset, train_raio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    전체 데이터셋에서 설정한 비율대로 train, val, test 나누는 함수
    """
    # 길이 계산
    total_len = len(dataset)
    train_len = int(total_len * train_raio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    
    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
    
    return train_dataset, val_dataset, test_dataset