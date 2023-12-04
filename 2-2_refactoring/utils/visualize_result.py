import matplotlib.pyplot as plt
import numpy as np

def visualize_result(model, data, labels, num_samples=5):
    """
    분류 모델의 결과를 시각화하는 함수.

    이 함수는 주어진 데이터의 샘플을 모델에 입력하고, 예측 결과를 시각적으로 표시한다.

    Parameters:
        model: 분류 모델.
        data (torch.Tensor): 입력 데이터.
        labels (torch.Tensor): 실제 레이블.
        num_samples (int): 시각화할 샘플의 수. 기본값은 5.
    """
    
    # 넘파이 기반 모델 예측 수행
    outputs = model.predict(data)
    predicted = np.argmax(outputs, axis=1)

    if num_samples is None or num_samples > len(data):
        num_samples = len(data)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        # 원본 이미지
        img = np.transpose(data[i], (1, 2, 0)) # 데이터 형태가 (C, H, W)인 경우
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original - Label: {}'.format(labels[i]))
        axes[i, 0].axis('off')

        # 예측 레이블
        axes[i, 1].text(0.5, 0.5, 'Predicted: {}'.format(predicted[i]), 
                        fontsize=14, ha='center')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# 사용 예시
# model: 학습된 분류 모델
# data: 테스트 데이터셋
# labels: 실제 레이블
# visualize_result(model, data, labels)
