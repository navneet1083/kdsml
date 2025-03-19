

class ModelSummary:
    def __init__(self, model, model_name="Model"):
        self.model = model
        self.model_name = model_name

    def get_inner_model(self):
        """
        If the model is wrapped in DistributedDataParallel (or similar),
        retrieve the underlying module. Also, if the model is a wrapper
        that contains a student model (e.g., FineTuneStudentModel), return that inner model.
        """
        if hasattr(self.model, "module"):
            self.model = self.model.module
        if hasattr(self.model, "student_model"):
            return self.model.student_model
        return self.model

    def model_summary(self):
        # If the model is wrapped (DDP), get the inner model.
        model = self.get_inner_model()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Try to count layers and get hidden size.
        num_layers = None
        hidden_size = None
        # For teacher models from Hugging Face (e.g., BertForSequenceClassification)
        if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
            num_layers = len(model.bert.encoder.layer)
            if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
                hidden_size = model.config.hidden_size
            elif hasattr(model.bert, "config") and hasattr(model.bert.config, "hidden_size"):
                hidden_size = model.bert.config.hidden_size
        # For our student model, we assume the model has attributes "embedding" and "layers"
        elif hasattr(model, "embedding") and hasattr(model, "layers"):
            num_layers = len(model.layers)
            hidden_size = model.embedding.embedding_dim

        print(f"{self.model_name} Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        if num_layers is not None:
            print(f"  Number of layers: {num_layers}")
        if hidden_size is not None:
            print(f"  Hidden size: {hidden_size}")


