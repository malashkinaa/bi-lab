from datetime import datetime
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    silhouette_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.decomposition import PCA

class Datamining:
    """Class to handle BI operations and generate BI tables."""

    def __init__(self, biDataset):
        """
        Initializes the Datamining with a biDataset instance.
        
        Args:
            biDataset (BiDataset): An instance of BiDataset containing BI data.
        """
        self.biDataset = biDataset
        self.tables = {}

    def classification_task(self):
        fact_loan = self.biDataset.tables['fact_loan']
        dim_client = self.biDataset.tables['dim_client']
        dim_account = self.biDataset.tables['dim_account']

        dim_client['client_age'] = dim_client['birth_number'].apply(self.extract_age)
        data = pd.merge(fact_loan, dim_client, on="client_id")
        data = pd.merge(data, dim_account, on="account_id")
        data.rename(columns={'district_id_x': 'loan_district', 'date_y': 'account_date'}, inplace=True)

        # calclate age based on birth_date
        # data['birth_date'] = pd.to_datetime(data['birth_date'])
        # data['age'] = (pd.to_datetime('today') - pd.to_datetime(data['birth_date'])).dt.days // 365

        print("Розпочато завдання класифікації статусу кредитів...")
        X = data[['amount', 'duration', 'payments', 'client_age', 'loan_district', 'account_date']]
        y = data['status']

        if X.empty:
            print("Немає даних для класифікації.")
            return pd.DataFrame([{"Model": "NoData", "Accuracy": 0, "F1 Score": 0}]), None, None

        if y.nunique() < 2:
            print("Недостатньо класів для класифікації.")
            return pd.DataFrame([{"Model": "NoData", "Accuracy": 0, "F1 Score": 0}]), None, None

        print("Розділення даних на тренувальні та тестові...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print("Навчання Logistic Regression...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lr = LogisticRegression(max_iter=2000, n_jobs=-1)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr, average='macro')

        print("Навчання Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf, average='macro')

        print("Класифікація статусу кредитів завершена.")

        results = {
            'Model': ['Logistic Regression', 'Random Forest Classifier'],
            'Accuracy': [round(accuracy_lr, 4), round(accuracy_rf, 4)],
            'F1 Score': [round(f1_lr, 4), round(f1_rf, 4)]
        }

        df_results = pd.DataFrame(results)
        return df_results, y_test, y_pred_rf
    
    def extract_age(self, birth_number, reference_year=2025):
        """Для 6- або 7-цифрового birth_number: YYMMDD.
        Якщо YY<=25 => 2000+YY, інакше => 1900+YY."""
        if pd.isna(birth_number):
            return 0
        try:
            s = str(int(birth_number))
            if len(s) < 2:
                return 0
            yy = int(s[:2])
            if yy <= 25:
                year = 2000 + yy
            else:
                year = 1900 + yy
            return reference_year - year
        except:
            return 0
    
    def clastering_task(self):
        fact_loan = self.biDataset.tables['fact_loan']
        fact_trans = self.biDataset.tables['fact_trans']
        dim_client = self.biDataset.tables['dim_client']

        # --- Step 1: Prepare age from birth_number ---
        dim_client["age"] = dim_client["birth_number"].apply(self.extract_age)

        # # --- Step 2: Extract unique clients from fact_trans and fact_loan ---
        # clients_from_trans = fact_trans[["client_id", "account_id"]].drop_duplicates()
        # clients_from_loan = fact_loan[["client_id", "account_id"]].drop_duplicates()

        # --- Step 3: Aggregate transaction data per client ---
        trans_agg = fact_trans.groupby("client_id").agg(
            total_transactions=("amount", "count"),
            avg_transaction_amount=("amount", "mean"),
            avg_balance=("balance", "mean")
        ).reset_index()

        # --- Step 4: Aggregate loan info per client ---
        loan_agg = fact_loan.groupby("client_id").agg(
            has_loan=("loan_id", lambda x: 1),
            avg_loan_amount=("amount", "mean")
        ).reset_index()

        # --- Step 5: Determine card ownership from fact_trans and fact_loan ---
        trans_cards = fact_trans[["client_id", "card_id"]].dropna().copy()
        loan_cards = fact_loan[["client_id", "card_id"]].dropna().copy()
        card_clients = pd.concat([trans_cards, loan_cards]).dropna().drop_duplicates()
        card_clients["has_card"] = 1
        card_flag = card_clients.groupby("client_id")["has_card"].max().reset_index()

        # --- Step 6: Merge everything into client-level features ---
        features = dim_client[["client_id", "age"]]
        features = features.merge(trans_agg, on="client_id", how="left")
        features = features.merge(loan_agg, on="client_id", how="left")
        features = features.merge(card_flag, on="client_id", how="left")

        # --- Step 7: Clean up missing values ---
        features["has_loan"] = features["has_loan"].fillna(0)
        features["has_card"] = features["has_card"].fillna(0)
        features = features.fillna(0)

        print("Підготовка даних для кластеризації завершена.")
        print("Перші 5 рядків підготовлених даних:")
        print(features.head())

        print("Розпочато завдання кластеризації клієнтів...")
        X = features[['age', 'total_transactions', 'avg_transaction_amount', 'avg_balance', 'has_loan', 'avg_loan_amount', 'has_card']]

        if X.empty:
            print("Немає даних для кластеризації.")
            return pd.DataFrame([{"Model": "NoData", "Silhouette Score": 0}]), None, None

        print("Навчання MiniBatchKMeans кластеризатора...")
        kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=10000)
        labels_km = kmeans.fit_predict(X)

        print("Обчислення Silhouette Score для MiniBatchKMeans...")
        silhouette_km = silhouette_score(X, labels_km)

        # print("Навчання Random Forest Classifier для кластеризації...")
        # rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        # rf.fit(X, labels_km)
        # labels_rf = rf.predict(X)

        # print("Обчислення Silhouette Score для Random Forest Classification...")
        # silhouette_rf = silhouette_score(X, labels_rf)

        # print("Навчання DBSCAN кластеризатора...")
        # db = DBSCAN(eps=0.5, min_samples=5)
        # labels_db = db.fit_predict(X)
        # print("Обчислення Silhouette Score для DBSCAN...")
        # silhouette_db = silhouette_score(X, labels_db)

        print("Навчання AgglomerativeClustering кластеризатора...")
        agg = AgglomerativeClustering(n_clusters=3)
        labels_agg = agg.fit_predict(X)
        print("Обчислення Silhouette Score для AgglomerativeClustering...")
        silhouette_agg = silhouette_score(X, labels_agg)

        print("Кластеризація клієнтів завершена.")

        results = {
            'Model': ['MiniBatchKMeans', 'AgglomerativeClustering'],
            'Clusters': [len(set(labels_km)), len(set(labels_agg))],
            'Silhouette Score': [round(silhouette_km, 4), round(silhouette_agg, 4)]
        }

        df_results = pd.DataFrame(results)
        return df_results, labels_km, labels_agg
    
    def forecasting_task(self):
        fact_trans = self.biDataset.tables['fact_trans']
        dim_client = self.biDataset.tables['dim_client']
        dim_account = self.biDataset.tables['dim_account']
        dim_district = self.biDataset.tables['dim_district']
        dim_date = self.biDataset.tables['dim_date']

        dim_client['client_age'] = dim_client['birth_number'].apply(self.extract_age)
        feature = pd.merge(fact_trans, dim_client, on="client_id")
        feature = pd.merge(feature, dim_account, on="account_id")
        feature = pd.merge(feature, dim_district, left_on="district_id_x", right_on="district_id")
        feature = pd.merge(feature, dim_date, left_on="date_x", right_on="date")
        feature.rename(columns={'district_id_x': 'trans_district','district_id_y': 'account_district', 'date_y': 'account_date', 'date_x': 'trans_date'}, inplace=True)

        agg = feature.groupby(['client_id', 'year']).agg({
        'amount': 'sum',              # total transaction volume
        'trans_id': 'count',          # number of transactions
        'balance': 'mean',            # average balance
        'frequency': 'last',         # transaction frequency
        'region': 'first',            # from district
        'client_age': 'first',      # will be used to get age
    }).reset_index().rename(columns={'trans_id': 'trans_count', 'amount': 'trans_amount'})


        print("Розпочато завдання прогнозу транзакцій...")
        X = agg[['trans_count', 'balance', 'client_age']] # add frequency, 'region',
        y = agg['trans_amount']

        if X.empty or y.empty:
            print("Немає даних для регресії.")
            return pd.DataFrame([{"Model": "NoData", "MSE": 0}]), None, None

        print("Розділення даних на тренувальні та тестові...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        print("Навчання Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        mse_lr = mean_squared_error(y_test, y_pred_lr)

        print("Навчання Random Forest Regressor...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)

        print("Регресія транзакцій завершена.")

        results = {
            'Model': ['Linear Regression', 'Random Forest Regressor'],
            'MSE': [round(mse_lr, 4), round(mse_rf, 4)]
        }

        df_results = pd.DataFrame(results)
        return df_results, y_test, y_pred_rf

    def processing(self):
        self.df_classification, self.y_test_class, self.y_pred_rf_class = self.classification_task()
        self.df_clustering, self.labels_km, self.labels_rf = self.clastering_task()
        self.df_forecasting, self.forecasting_test, self.forecasting_rf = self.forecasting_task()
        