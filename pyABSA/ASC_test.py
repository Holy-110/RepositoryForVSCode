from transformers import pipeline

# 以前のKeras/TensorFlow関連のエラーを回避するため、framework="pt"を指定します
# DeBERTaベースのABSAモデルをロード
classifier = pipeline(
    "text-classification", 
    model="yangheng/deberta-v3-base-absa-v1.1",
    framework="pt"
)

# テスト文
sentence = "サービスは遅かったが、食事はおいしかった"

# アスペクト「food」に対する感情を推論
result_food = classifier(sentence, text_pair="食事")
print(result_food)

# アスペクト「service」に対する感情を推論
result_service = classifier(sentence, text_pair="サービス")
print(result_service)