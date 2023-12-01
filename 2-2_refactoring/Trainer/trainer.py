class Trainer:
    
    """신경망 훈련을 대신 해주는 클래스"""
    
    def __init__(self, network, x_train_loader, x_test_loader,
                 epochs=30, mini_batch_size=32,
                 optimizer='adagrad', optimizer_param={'lr':0.0001}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.train_loader = x_train_loader
        self.test_loader = x_test_loader
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = len(x_train) # x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.epochs = epochs
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.train_mode = True

    def train_step(self):
            
        dataloader = self.train_loader if self.train_mode else self.test_loader
        name = "train" if self.train_mode else "evaluate"

        for x_batch, t_batch in dataloader:
            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)
            loss = self.network.loss(x_batch, t_batch)
            
            if self.train_mode:
                self.train_loss_list.append(loss)
                
            if self.verbose: print(f"\t{name} loss: {loss}")

            if self.current_iter % self.iter_per_epoch == 0:
                self.current_epoch += 1
    
                x_train_sample, t_train_sample = x_batch, t_batch
                train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            
            if self.train_mode:
                self.train_acc_list.append(train_acc)
            else:
                self.test_acc_list.append(test_acc)

                if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
            self.current_iter += 1
            

    def train(self, current_epochs):
        for epoch in range(self.epochs):
            self.train_step()
            self.network.save_params(file_name=f"epoch_{epoch+current_epochs+1}.pkl")
            print(f"model({epoch+1}/{self.epochs}) is saved!")
            graph(self.train_loss_list, 'loss', 'red', f"epoch_{epoch+current_epochs+1}")

    def test(self, test_data_length = len(test_data)):
        hits = 0.0
        acc = 0.0
        for imgs, labels in self.test_loader:
            hits += self.network.accuracy(imgs, labels)
        acc = hist/test_data_length

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))