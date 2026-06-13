class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
            return True  # improved
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
            return False