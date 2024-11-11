import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from traffic_attributes import TrafficAttr


SAMPLED_PATH = './201501_sampled.txt'   # SVMのトレーニングに使用するサンプリングデータ (2015年1月)
# SAMPLED_PATH = './201501_15-31_sampled.txt'   # SVMのトレーニングに使用するサンプリングデータ (2015年1月15日から31日まで)
# TEST_PATH = './201502_sampled.txt'    # SVMのテストに使用するサンプリングデータ (2015年2月)
TEST_PATH = './20150201_sampled.txt'    # SVMのテストに使用するサンプリングデータ (2015年2月1日)

# 使用する数値データの特徴量
NUMERIC_FEATURES = [
    TrafficAttr.DURATION.name,
    TrafficAttr.SOURCE_BYTES.name,
    TrafficAttr.DESTINATION_BYTES.name,
    TrafficAttr.COUNT.name,
    TrafficAttr.SAME_SRV_RATE.name,
    TrafficAttr.SERROR_RATE.name,
    TrafficAttr.SRV_SERROR_RATE.name,
    TrafficAttr.DST_HOST_COUNT.name,
    TrafficAttr.DST_HOST_SRV_COUNT.name,
    TrafficAttr.DST_HOST_SAME_SRC_PORT_RATE.name,
    TrafficAttr.DST_HOST_SERROR_RATE.name,
    TrafficAttr.DST_HOST_SRV_SERROR_RATE.name,
]

# 使用するカテゴリカルデータの特徴量
CATEGORICAL_FEATURES = [
    TrafficAttr.IDS_DETECTION.name,
    TrafficAttr.MALWARE_DETECTION.name,
    TrafficAttr.ASHULA_DETECTION.name,
    # TrafficAttr.SERVICE.name,
    # TrafficAttr.FLAG.name,
    # TrafficAttr.PROTOCOL.name,
]


def main():
    # トレーニングデータとテストデータの読み込みと前処理
    X_train, y_train = load_and_preprocess_data(SAMPLED_PATH)
    X_test, y_test = load_and_preprocess_data(TEST_PATH)

    # トレーニングデータとテストデータの特徴量を一致させる
    # すべての特徴量が両方のデータセットに存在するようにし、不足している特徴量には0を埋める
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    # トレーニングデータとテストデータを標準化する
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVMモデルの定義と学習
    # RBFカーネルを使用し、ハイパーパラメータCとgammaを設定してモデルをトレーニングデータで学習させる
    # - kernel: 'rbf'はガウシアンカーネル（RBFカーネル）を使用することを指定
    # - C: 誤分類をどれだけ許容するかを制御するパラメータ。大きい値にすると誤分類を減らすが、過学習のリスクが増える
    # - gamma: RBFカーネルの幅を制御するパラメータ。大きい値にするとモデルがより複雑になり、過学習のリスクが増える
    # - random_state: 結果の再現性を確保するための乱数シード
    # - max_iter: 最大反復回数。収束しない場合の停止条件
    model = SVC(kernel='rbf', C=100, gamma=0.1, random_state=42, max_iter=5000)
    model.fit(X_train, y_train)

    # テストデータでの評価
    y_pred = model.predict(X_test)  # テストデータを使って予測を行う
    accuracy = accuracy_score(y_test, y_pred)   # 予測の正確さを計算
    print(f'Accuracy: {accuracy:.5f}')  # 精度を表示
    print(classification_report(y_test, y_pred))    # 適合率、再現率、F1スコアを表示


def load_and_preprocess_data(file_path):
    '''
    指定されたファイルパスからデータを読み込み、前処理を行う関数。

    前処理の手順:
    1. データを読み込む。
    2. 各ラベルのサンプル数を表示する。
    3. 必要な特徴量のみを選択する。
    4. カテゴリカルデータをダミー変数に変換する。
    5. 特徴量とラベルに分割する。

    引数:
    file_path (str): データファイルのパス。

    戻り値:
    X (DataFrame): 特徴量データ。
    y (Series): ラベルデータ。
    '''
    data = pd.read_csv(file_path, delimiter='\t')

    # 各LABELのサンプル数を表示
    print(file_path)
    print(data[TrafficAttr.LABEL.name].value_counts(), end='\n\n')

    # 必要な特徴量のみを選択
    required_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TrafficAttr.LABEL.name]
    data = data[required_features]

    # カテゴリカルデータをダミー変数に変換
    data = pd.get_dummies(data, columns=CATEGORICAL_FEATURES, drop_first=True)

    # 特徴量とラベルに分割
    y = data[TrafficAttr.LABEL.name]
    X = data.drop(columns=[TrafficAttr.LABEL.name])

    return X, y


if __name__ == '__main__':
    main()
