from pyabsa import AspectTermExtraction as ATEPC

# 推論に使用する文章
sentence = "The food was exceptional, although the service was a bit slow."

# 1. モデルのロード (E2E ATEPCタスク用)
# ATEPC.AspectExtractorを使用し、事前学習済みモデルをロードします。
# pyabsaは、指定されたチェックポイントをATEPCモデルとして自動的に構成します。
try:
    aspect_extractor = ATEPC.AspectExtractor(
        #'yangheng/deberta-v3-base-absa-v1.1', # 使用するDeBERTaベースのABSAモデル
        'koheiduck/bert-japanese-finetuned-sentiment',
        auto_device=True,                      # GPUまたはCPUを自動で選択
        cal_perplexity=False                   # パフォーマンスのため、計算負荷の高いPerplexity計算はOFF
    )

    # 2. 推論の実行
    # predictメソッドにテキストのリストを渡し、E2Eでアスペクトと感情を抽出します。
    result = aspect_extractor.predict(
        [sentence],
        print_result=True,     # 結果をコンソールに出力
        save_result=False,     # 結果をファイルに保存しない
        ignore_error=True,     
        pred_sentiment=True    # 抽出されたアスペクトに対して感情も予測
    )

    # 3. 結果の出力と確認
    print("\n--- E2E 推論結果 ---")
    if result and result[0]:
        # 結果はリストで返されるため、最初の要素を出力します
        print(result[0])
    else:
        print("推論結果が得られませんでした。")

except Exception as e:
    print(f"\nエラーが発生しました。ライブラリのバージョンを確認してください: {e}")