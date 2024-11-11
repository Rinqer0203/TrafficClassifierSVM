import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from traffic_attributes import TrafficAttr


SAMPLED_PATH = './201501_sampled.txt'   # SVMのトレーニングに使用するサンプリングデータ (2015年1月)
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
    # TrafficAttr.IDS_DETECTION.name,
    # TrafficAttr.MALWARE_DETECTION.name,
    # TrafficAttr.ASHULA_DETECTION.name,
    TrafficAttr.SERVICE.name,
    TrafficAttr.FLAG.name,
    TrafficAttr.PROTOCOL.name,
]


def main():
    # トレーニングデータとテストデータの読み込みと前処理
    X_train, y_train = load_and_preprocess_data(SAMPLED_PATH)
    X_test, y_test = load_and_preprocess_data(TEST_PATH)

    # トレーニングデータとテストデータの特徴量を一致させる
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    # データの標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVMモデルの定義と学習
    model = SVC(kernel='rbf', C=100, gamma=0.1, random_state=42, max_iter=5000)
    model.fit(X_train, y_train)

    # テストデータでの評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred))


def load_and_preprocess_data(file_path):
    '''
    データを読み
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
