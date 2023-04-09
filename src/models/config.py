class Config:
    def __init__(self) -> None:
        self.dir = "/home/naer/work/ViDesc/src/"
        self.videos_path = "/datasets/mixkit/"
        self.data_csv = "/datasets/ru_mixkit_train.csv"
        self.encoder_backbone = "microsoft/xclip-base-patch16-16-frames"
        self.decoder_backbone = "sberbank-ai/rugpt3small_based_on_gpt2"
        self.train_features_path = "/datasets/mixkit_features_train.pkl"
        self.test_features_path = "/datasets/mixkit_features_test.pkl"
        self.out_dir = "src/models/checkpoints/"
        self.model_path = "src/models/checkpoints/ViDesc-009.pt"
        self.epochs = 10
        self.extract_size = 224, 224
        self.save_every = 3
        self.prefix_length = 35
        self.prefix_size = 512
        self.bs = 8
        self.only_prefix = False
        self.lr = 5e-2
        self.warmup_steps = 5000
        self.prompt = "Описание видео: "
        self.max_words = 50
        self.temperature = 1.0
        self.top_p = 0.98
