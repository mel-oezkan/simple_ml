import torch

from vit.models.classifier import CNNClassifier, MLPClassifier, load_cnn_classifier, load_mlp_classifier


def test_load_cnn_classifier_supports_training_checkpoint(tmp_path):
    original = CNNClassifier(dropout=0.2)
    legacy_state_dict = dict(original.state_dict())
    legacy_state_dict["classifier.4.weight"] = legacy_state_dict.pop(
        "out_head.weight"
    )
    legacy_state_dict["classifier.4.bias"] = legacy_state_dict.pop("out_head.bias")
    checkpoint_path = tmp_path / "classifier.pt"
    torch.save(
        {
            "hyperparameters": {"dropout": 0.2},
            "model_state_dict": legacy_state_dict,
        },
        checkpoint_path,
    )

    loaded = load_cnn_classifier(checkpoint_path)
    logits, features = loaded(
        torch.randn(2, 1, 28, 28),
        return_features=True,
    )

    assert logits.shape == (2, 10)
    assert features.shape == (2, loaded.feature_size)
    assert torch.equal(loaded.out_head.weight, original.out_head.weight)


def test_load_mlp_classifier(tmp_path):
    original = MLPClassifier(num_classes=10, dropout=0.1)
    checkpoint_path = tmp_path / "mlp.pt"
    torch.save(
        {"hyperparameters": {"dropout": 0.1}, "model_state_dict": original.state_dict()},
        checkpoint_path,
    )
    
    model = load_mlp_classifier(
        checkpoint_path, num_classes=10, map_location="cpu"
    )

    model.eval()

    # test the forward step
    logits = model(torch.zeros((1, 28, 28)))

def test_classifier_on_train():
    # classifier path

    pass