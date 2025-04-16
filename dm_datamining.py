import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    silhouette_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

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
        """Для 6- або 7-цифрового birth_number: YYMMDD. Якщо YY<=25 => 2000+YY, інакше => 1900+YY."""
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

        # --- Step 2: Aggregate transaction data per client ---
        trans_agg = fact_trans.groupby("client_id").agg(
            total_transactions=("amount", "count"),
            avg_transaction_amount=("amount", "mean"),
            avg_balance=("balance", "mean")
        ).reset_index()

        # --- Step 3: Aggregate loan info per client ---
        loan_agg = fact_loan.groupby("client_id").agg(
            has_loan=("loan_id", lambda x: 1),
            avg_loan_amount=("amount", "mean")
        ).reset_index()

        # --- Step 4: Determine card ownership from fact_trans and fact_loan ---
        trans_cards = fact_trans[["client_id", "card_id"]].dropna().copy()
        loan_cards = fact_loan[["client_id", "card_id"]].dropna().copy()
        card_clients = pd.concat([trans_cards, loan_cards]).dropna().drop_duplicates()
        card_clients["has_card"] = 1
        card_flag = card_clients.groupby("client_id")["has_card"].max().reset_index()

        # --- Step 5: Merge everything into client-level features ---
        features = dim_client[["client_id", "age"]]
        features = features.merge(trans_agg, on="client_id", how="left")
        features = features.merge(loan_agg, on="client_id", how="left")
        features = features.merge(card_flag, on="client_id", how="left")

        # --- Step 6: Clean up missing values ---
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
        'frequency': 'last',         # account frequency status
        'region': 'first',            # from district
        'client_age': 'first',      # will be used to get age
    }).reset_index().rename(columns={'trans_id': 'trans_count', 'amount': 'trans_amount'})

        # convert strings to numbers because models like LinearRegression, RandomForest, SVC, etc. do not support non-numeric (string) values in X
        le = LabelEncoder()
        agg['frequency_encoded'] = le.fit_transform(agg['frequency'])
        agg['region_encoded'] = le.fit_transform(agg['region'])

        print("Розпочато завдання прогнозу транзакцій...")
        print("Розділення даних на тренувальні та тестові...")
        # Filter training and test data
        past = agg[agg['year'] < 1998] # training past data
        present = agg[agg['year'] == 1998] # test data for 1998

        future = agg[agg['year'] == 1998].copy() # future data for 1999, based on their 1998 behavior
        future['year'] = 1999  # simulate next year in the copy

        if past.empty or present.empty:
            print("Немає даних для прогнозу.")
            return pd.DataFrame([{"Model": "NoData", "MSE": 0}]), None, None
        
        X_train = past[['trans_count', 'balance', 'client_age', 'frequency_encoded', 'region_encoded']]
        Y_train = past['trans_amount']

        X_test = present[['trans_count', 'balance', 'client_age', 'frequency_encoded', 'region_encoded']]
        Y_test = present['trans_amount']

        # use the same features for real future prediction
        X_future = future[['trans_count', 'balance', 'client_age', 'frequency_encoded', 'region_encoded']]  
        

        print("Навчання Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, Y_train) # train the model on the past data
        y_pred_lr = lr.predict(X_test) # predict the test in present year
        r2_lr = r2_score(Y_test, y_pred_lr)
        future['lr_trans_amount'] = lr.predict(X_future)  # predict the real future

        print("Навчання Random Forest Regressor...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, Y_train)
        y_pred_rf = rf.predict(X_test)
        r2_rf = r2_score(Y_test, y_pred_rf)
        future['rf_trans_amount'] = rf.predict(X_future)

        print("Прогноз транзакцій завершено.")

        results = {
            'Model': ['Linear Regression', 'Random Forest Regressor'],
            'R2 Score': [round(r2_lr, 4), round(r2_rf, 4)]
        }

        df_results = pd.DataFrame(results)
        return df_results, Y_test, y_pred_rf

    # Do clients with high transaction volumes tend to pay off loans more reliably? 
    def dependencies_task(self):
        fact_loan = self.biDataset.tables['fact_loan']
        fact_trans = self.biDataset.tables['fact_trans']
        dim_client = self.biDataset.tables['dim_client']

        # --- Step 1: Aggregate transaction data per client ---
        trans_agg = fact_trans.groupby("client_id").agg(
            total_transactions=("amount", "count"),
            avg_balance=("balance", "mean")
        ).reset_index()

        # --- Step 2: Aggregate loan info per client ---
        # A = fully paid
        # B = partially paid
        # C = late
        # D = defaulted
        status_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        fact_loan['status_number'] = fact_loan['status'].map(status_map)
        loan_agg = fact_loan.groupby("client_id").agg(
            has_loan=("loan_id", lambda x: 1),
            loan_status=("status_number", "mean") # overall reliability score
        ).reset_index()

        # --- Step 3: Merge 3 tables into a table called features ---
        features = dim_client[["client_id", "age"]]
        features = features.merge(trans_agg, on="client_id", how="left")
        features = features.merge(loan_agg, on="client_id", how="left")

        # --- Step 4: Fill missing values with 0---
        features["total_transactions"] = features["total_transactions"].fillna(0)
        features["avg_balance"] = features["avg_balance"].fillna(0)
        features = features.fillna(0)

        X = features[['total_transactions', 'avg_balance']] 
        Y = features['loan_status']

        if X.empty:
            print("Немає даних для виявлення залежностей.")
            return pd.DataFrame([{"Model": "NoData", "Accuracy": 0, "F1 Score": 0}]), None, None

        if Y.nunique() < 2:
            print("Недостатньо класів для виявлення залежностей.")
            return pd.DataFrame([{"Model": "NoData", "Accuracy": 0, "F1 Score": 0}]), None, None

        print("Розділення даних на тренувальні та тестові...")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42, stratify=Y
        )

        print("Навчання LinearSVC ...")
        # Use a pipeline to ensure the features are scaled (important for SVMs)
        svc = make_pipeline(StandardScaler(), LinearSVC(random_state=42, max_iter=10000))
        svc.fit(X_train, Y_train) # Train the model on 70%
        y_pred_svc = svc.predict(X_test) # Asking the already trained model to predict the remaining 30%
        accuracy_svc = accuracy_score(Y_test, y_pred_svc)
        f1_svc = f1_score(Y_test, y_pred_svc, average='weighted')

        print("Навчання Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, Y_train)
        y_pred_rf = rf.predict(X_test)
        accuracy_rf = accuracy_score(Y_test, y_pred_rf)
        f1_rf = f1_score(Y_test, y_pred_rf, average='weighted')

        print("Виявлення залежностей завершено.")

        results = {
            'Model': ['LinearSVC', 'Random Forest Classifier'],
            'Accuracy': [round(accuracy_svc, 4), round(accuracy_rf, 4)],
            'F1 Score': [round(f1_svc, 4), round(f1_rf, 4)]
        }

        df_results = pd.DataFrame(results)
        return df_results, Y_test, y_pred_rf

    def processing(self):
        self.df_classification, self.y_test_class, self.y_pred_rf_class = self.classification_task()
        self.df_clustering, self.labels_km, self.labels_rf = self.clastering_task()
        self.df_forecasting, self.forecasting_test, self.forecasting_rf = self.forecasting_task()
        self.df_dependencies, self.dependencies_test, self.dependencies_rf = self.dependencies_task()

