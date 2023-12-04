import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer.adam import Adam
from utils import graph, visualize_result
import wandb
from tqdm import tqdm

class Trainer:
    """
    신경망 모델의 훈련을 관리하는 클래스.

    이 클래스는 신경망 모델의 훈련을 위한 여러 기능을 제공한다. 
    훈련 데이터와 테스트 데이터에 대한 로딩, 모델의 업데이트, 
    훈련과 검증 과정의 정확도 계산 등을 수행합니다.

    Parameters:
        network: 훈련할 신경망 모델.
        x_train_loader: 훈련 데이터 로더.
        x_test_loader: 테스트 데이터 로더.
        epochs (int): 에폭 수.
        mini_batch_size (int): 미니 배치 크기.
        optimizer (str): 최적화 알고리즘의 이름.
        optimizer_param (dict): 최적화 알고리즘의 매개변수.
        evaluate_sample_num_per_epoch (int): 에폭당 평가할 샘플 수.
        verbose (bool): 상세 정보 출력 여부.
    """

    def __init__(self, network, x_train_loader, x_test_loader,
                 epochs=30, mini_batch_size=32,
                 optimizer='adam', optimizer_param={'lr':0.0001}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        
        # 네트워크와 데이터 로더 초기화
        self.network = network
        self.train_loader = x_train_loader
        self.test_loader = x_test_loader
        
        # 훈련 설정 초기화
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.verbose = verbose

        # 최적화 알고리즘 초기화
        optimizer_class_dict = {'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        # 훈련 데이터 크기 및 에폭당 반복 횟수 설정
        self.train_size = len(x_train_loader.dataset) # x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.current_iter = 0
        self.current_epoch = 0
        
        # 훈련 및 검증 과정의 기록을 저장하는 리스트
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        
        # 현재 학습중인지에 대한 상태
        self.train_mode = True


    def train_step(self):
        """
        하나의 에폭 동안 모델을 훈련하는 단계.
        """
        
        # 현재 훈련 상태에 따라 적절한 데이터 로더 선택
        dataloader = self.train_loader if self.train_mode else self.test_loader
        name = "train" if self.train_mode else "evaluate"

        # tqdm을 이용하여 진행 상황 시각화
        for x_batch, t_batch in tqdm(dataloader, desc=name):
            # 그래디언트 계산 및 최적화
            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)
            
            # 손실 계산
            loss = self.network.loss(x_batch, t_batch)
            
            # 훈련 중 손실 기록 및 wandb 로깅
            if self.train_mode:
                self.train_loss_list.append(loss)
                wandb.log({"train_loss": loss})
            
            # 상세 정보 출력    
            if self.verbose: 
                print(f"\t{name} loss: {loss}")

            # 에폭의 마지막에서 정확도 계산
            if self.current_iter % self.iter_per_epoch == 0:
                self.current_epoch += 1

                if self.train_mode:
                    train_acc = self.network.accuracy(x_batch, t_batch)
                    self.train_acc_list.append(train_acc)
                    wandb.log({"train_accuracy": train_acc})
                else:
                    test_acc = self.calculate_accuracy(self.test_loader)
                    self.test_acc_list.append(test_acc)
                    wandb.log({"test_accuracy": test_acc})

                    if self.verbose: 
                        print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
            self.current_iter += 1


    def train(self, current_epochs):
        """
        신경망 모델을 훈련하는 메소드.

        Args:
            current_epochs (int): 현재까지 완료된 에폭 수.
        """
        for epoch in range(self.epochs):
            self.train_step()
            
            if (epoch + 1) % 5 ==  0:
                test_data, test_labels = next(iter(self.test_loader))
                visualize_result.visualize_result(self.network, test_data, test_labels)
            
            # 모델 파라미터 저장 및 그래프 그리기
            self.network.save_params(file_name=f"epoch_{epoch+current_epochs+1}.pkl")
            print(f"model({epoch+1}/{self.epochs}) is saved!")
            graph(self.train_loss_list, 'loss', 'red', f"epoch_{epoch+current_epochs+1}")


    def test(self):
        """
        신경망 모델의 테스트 데이터에 대한 정확도를 평가하는 메소드.
        """
        hits = 0.0
        # 테스트 데이터 로더를 통해 정확도 계산
        for imgs, labels in self.test_loader:
            hits += self.network.accuracy(imgs, labels)
        test_acc = hits / len(self.test_loader.dataset)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


    def calculate_accuracy(self, loader):
        """
        데이터 로더를 통해 데이터셋의 정확도를 계산하는 메소드.

        Args:
            loader: 데이터 로더.

        Returns:
            float: 계산된 정확도.
        """
        hits = 0.0
        for imgs, labels in loader:
            hits += self.network.accuracy(imgs, labels)
        return hits / len(loader.dataset)