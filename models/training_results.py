class TrainingResults:
    """
    Stores the results and metrics from training.
    """

    def __init__(self):
        self.mean_MT_error = []  # mean manifold traversal denoising error over time
        self.mean_data_error = []  # mean baseline (noisy vs clean) error over time
        self.all_MT_errors = []  # individual MT errors for each sample
        self.training_times = []  # training time for each batch
        self.samples_processed = 0  # total number of samples processed

    def update(self, mt_error, data_error):
        """Update results with errors for current sample."""
        self.all_MT_errors.append(mt_error)

        if self.samples_processed == 0:
            self.mean_MT_error.append(mt_error)
            self.mean_data_error.append(data_error)
        else:
            # running average update
            n = self.samples_processed + 1
            prev_mt_mean = self.mean_MT_error[-1]
            prev_data_mean = self.mean_data_error[-1]

            new_mt_mean = ((n - 1) / n) * prev_mt_mean + (1 / n) * mt_error
            new_data_mean = ((n - 1) / n) * prev_data_mean + (1 / n) * data_error

            self.mean_MT_error.append(new_mt_mean)
            self.mean_data_error.append(new_data_mean)

        self.samples_processed += 1
