import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from traffic_attributes import TrafficAttr

# データのパス（フォルダでもファイルでも可）
SAMPLED_PATH = './201501_sampled.txt'
TEST_PATH = './201502_sampled.txt'

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

# 使用するカテゴリカルデータの特徴量（空でも可）
CATEGORICAL_FEATURES = [
    TrafficAttr.IDS_DETECTION.name,
    TrafficAttr.MALWARE_DETECTION.name,
    TrafficAttr.ASHULA_DETECTION.name,
    TrafficAttr.SERVICE.name,
    TrafficAttr.FLAG.name,
    TrafficAttr.PROTOCOL.name,
]


def main():
    X_train, y_train = load_and_preprocess_data(SAMPLED_PATH)
    X_test, y_test = load_and_preprocess_data(TEST_PATH)

    # トレーニングデータとテストデータの特徴量を一致させる
    X_train, X_test = align_features(X_train, X_test)
    X_train, X_test, _ = scale_data(X_train, X_test)

    # SVMモデルの定義と学習
    model = SVC(kernel='rbf', C=100, gamma=0.1, random_state=42, max_iter=5000)
    model.fit(X_train, y_train)

    # モデルの評価
    evaluate_model(model, X_test, y_test)


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t')

    # 各LABELのサンプル数を表示
    print(file_path)
    print(data[TrafficAttr.LABEL.name].value_counts(), end='\n\n')

    # 不要な特徴量を削除
    all_features = set(TrafficAttr.get_attribute_name_list())
    required_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TrafficAttr.LABEL.name])
    drop_features = list(all_features - required_features)
    data = data.drop(columns=drop_features)

    # カテゴリカルデータをダミー変数に変換
    data = pd.get_dummies(data, columns=CATEGORICAL_FEATURES, drop_first=True)

    y = data[TrafficAttr.LABEL.name]
    X = data.drop(columns=[TrafficAttr.LABEL.name])

    return X, y


def align_features(X_train, X_test):
    # トレーニングデータとテストデータの特徴量を一致させる
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)
    return X_train, X_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
