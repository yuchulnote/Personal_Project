import os
import shutil
import random

def split_dataset(src_folder, dst_base_folder, ratios=[0.8, 0.1, 0.1]):
    """
    지정된 소스 폴더(src_folder)에서 파일을 읽어와서, 
    훈련(train), 검증(val), 테스트(test) 폴더로 분할하는 함수.

    이 함수는 소스 폴더의 하위 폴더 구조를 유지하며, 지정된 비율에 따라 파일을 무작위로 분할한다.

    Parameters:
        src_folder (str): 소스 폴더의 경로.
        dst_base_folder (str): 대상 폴더의 기본 경로. 이 폴더 내에 train, val, test 폴더가 생성됨.
        ratios (list): 훈련, 검증, 테스트 데이터셋으로 분할할 비율의 리스트. 기본값은 [0.8, 0.1, 0.1].
    """
    
    # 대상 폴더의 경로 생성
    dst_folders = [os.path.join(dst_base_folder, name) for name in ["train", "val", "test"]]

    # 대상 폴더가 존재하지 않는 경우 생성
    for folder in dst_folders:
        os.makedirs(folder, exist_ok=True)

    # 소스 폴더의 모든 하위 폴더 검색
    sub_folders = [f.path for f in os.scandir(src_folder) if f.is_dir()]

    # 각 하위 폴더의 이미지를 처리
    for sub in sub_folders:
        sub_folder_name = os.path.basename(sub)  # 하위 폴더 이름 추출
        images = [img for img in os.listdir(sub) if img.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)  # 이미지 리스트를 무작위로 섞음

        # 비율 리스트 복제 (원본 수정 방지)
        temp_ratios = ratios.copy()

        # 각 이미지를 무작위로 선택된 폴더에 복사
        for img_file in images:
            selected_folder = random.choices(dst_folders, weights=temp_ratios, k=1)[0]

            # 선택된 폴더의 비율 조정
            idx = dst_folders.index(selected_folder)
            temp_ratios[idx] -= 1 / len(images)

            # 최종 대상 폴더에 동일한 하위 폴더 구조 생성
            final_destination = os.path.join(selected_folder, sub_folder_name)
            os.makedirs(final_destination, exist_ok=True)

            # 이미지 파일을 최종 대상 폴더로 복사
            shutil.copy(os.path.join(sub, img_file), os.path.join(final_destination, img_file))
