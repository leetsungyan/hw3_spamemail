from src.inference import predict_text, decision_to_confidence


class DummyModel:
    def predict(self, X):
        return [1]

    def decision_function(self, X):
        return [2.0]


def test_decision_to_confidence():
    assert 0.0 <= decision_to_confidence(0.0) <= 1.0
    assert decision_to_confidence(2.0) > 0.5


def test_predict_text_with_dummy_model():
    model = DummyModel()
    res = predict_text(model, "You have won a prize")
    assert isinstance(res, dict)
    assert res["label"] == "spam"
    assert 0.0 <= res["confidence"] <= 1.0
