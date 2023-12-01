class Trainer:
    """
    신경망 모델의 훈련을 관리하는 클래스.

    이 클래스는 신경망 모델의 훈련을 위한 여러 기능을 제공합니다. 
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
        self.network = network
        self.verbose = verbose
        self.train_loader = x_train_loader
        self.test_loader = x_test_loader
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # 최적화 알고리즘 설정
        optimizer_class_dict = {'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        # 훈련 데이터 크기 및 에폭당 반복 횟수 설정
        self.train_size = len(x_train_loader.dataset) # x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.epochs = epochs
        self.current_iter = 0
        self.current_epoch = 0
        
        # 훈련 및 검증 과정의 기록
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.train_mode = True

    def train_step(self):
        """
        하나의 에폭 동안 모델을 훈련하는 단계.
        """
        dataloader = self.train_loader if self.train_mode else self.test_loader
        name = "train" if self.train_mode else "evaluate"

        for x_batch, t_batch in dataloader:
            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)
            loss = self.network.loss(x_batch, t_batch)
            
            if self.train_mode:
                self.train_loss_list.append(loss)
                
            if self.verbose: 
                print(f"\t{name} loss: {loss}")

            if self.current_iter % self.iter_per_epoch == 0:
                self.current_epoch += 1

                train_acc = self.network.accuracy(x_batch, t_batch)
                if self.train_mode:
                    self.train_acc_list.append(train_acc)
                else:
                    self.test_acc_list.append(test_acc)

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
            self.network.save_params(file_name=f"epoch_{epoch+current_epochs+1}.pkl")
            print(f"model({epoch+1}/{self.epochs}) is saved!")
            graph(self.train_loss_list, 'loss', 'red', f"epoch_{epoch+current_epochs+1}")

    def test(self):
        """
        신경망 모델의 테스트 데이터에 대한 정확도를 평가하는 메소드.
        """
        hits = 0.0
        acc = 0.0
        for imgs, labels in self.test_loader:
            hits += self.network.accuracy(imgs, labels)
        acc = hits / len(self.test_loader.dataset)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
