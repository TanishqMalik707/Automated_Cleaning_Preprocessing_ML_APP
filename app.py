import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import io
import joblib
import base64
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(layout="wide")
st.title("🧹 Advanced Data Cleaning & ML App")

df = None
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    st.sidebar.header("⚙️ Options")

    # ----------------------------- Null Value Check & Removal
    with st.sidebar.expander("🚫 Null Value Handling", expanded=True):
        if df.isnull().sum().sum() > 0:
            st.write("🔍 Columns with Null Values:")
            st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])
            remove_nulls = st.multiselect("Select columns to remove rows with nulls", df.columns[df.isnull().any()])
            if st.button("Remove Null Rows"):
                df.dropna(subset=remove_nulls, inplace=True)
                st.success("Selected null rows removed.")
        else:
            st.info("✅ No missing values detected!")

    # ----------------------------- Data Type Conversion
    with st.sidebar.expander("🔧 Data Type Conversion"):
        st.write("Convert selected columns to different data types:")

        col_to_convert = st.multiselect("Select columns to convert", df.columns)
        conversion_type = st.selectbox("Convert to", ["int", "float", "str", "datetime", "bool", "category", "Auto Detect"])

        if st.button("Apply Conversion"):
            for col in col_to_convert:
                try:
                    if conversion_type == "datetime":
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif conversion_type == "bool":
                        df[col] = df[col].astype(bool)
                    elif conversion_type == "category":
                        df[col] = df[col].astype("category")
                    elif conversion_type == "Auto Detect":
                        # Try auto-detecting datetime
                        if df[col].dtype == 'object' and df[col].str.match(r'\d{4}-\d{2}-\d{2}').any():
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            st.success(f"{col} converted to datetime (auto)")
                        else:
                            st.warning(f"Could not auto-detect type for {col}")
                    else:
                        df[col] = df[col].astype(conversion_type)
                    st.success(f"Converted {col} to {conversion_type}")
                except Exception as e:
                    st.error(f"Error converting {col}: {e}")

    # ----------------------------- Imputation
    with st.sidebar.expander("👭 Imputation"):
        impute_cols = st.multiselect("Select columns for imputation", df.columns)
        impute_strategy = st.selectbox("Strategy", ["mean", "median", "most_frequent"])
        if st.button("Apply Imputation"):
            imputer = SimpleImputer(strategy=impute_strategy)
            df[impute_cols] = imputer.fit_transform(df[impute_cols])

    # ----------------------------- Outlier Removal
    with st.sidebar.expander("❌ Outlier Removal (IQR)"):
        outlier_cols = st.multiselect("Select numeric columns", df.select_dtypes(include=np.number).columns)
        if st.button("Remove Outliers"):
            for col in outlier_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # ----------------------------- Transformations
    with st.sidebar.expander("🔄 Transformations"):
        transform_cols = st.multiselect("Select columns for transformation", df.select_dtypes(include=np.number).columns)
        transform_type = st.selectbox("Transformation", ["Log", "Square Root", "PowerTransformer"])
        if st.button("Apply Transformation"):
            try:
                for col in transform_cols:
                    if transform_type == "Log":
                        df[col] = np.log1p(df[col])
                    elif transform_type == "Square Root":
                        df[col] = np.sqrt(df[col])
                    else:
                        pt = PowerTransformer()
                        df[transform_cols] = pt.fit_transform(df[transform_cols])
                        break
            except Exception as e:
                st.error(f"Transformation failed: {e}")

    # ----------------------------- Encoding
    with st.sidebar.expander("🎨 Encoding Categorical Columns"):
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            selected_cat = st.multiselect("Select categorical columns", cat_cols)
            encoding = st.radio("Encoding method", ["Label Encoding", "One-Hot Encoding"])
            if st.button("Apply Encoding"):
                for col in selected_cat:
                    if encoding == "Label Encoding":
                        df[col] = LabelEncoder().fit_transform(df[col])
                    else:
                        df = pd.get_dummies(df, columns=[col])
        else:
            st.info("No categorical columns found.")

    # ----------------------------- Scaling
    with st.sidebar.expander("📏 Scaling"):
        scale_cols = st.multiselect("Select numeric columns to scale", df.select_dtypes(include=np.number).columns)
        scaler_option = st.selectbox("Select Scaler", ["None", "StandardScaler", "MinMaxScaler"])
        if scaler_option != "None" and st.button("Apply Scaling"):
            scaler = StandardScaler() if scaler_option == "StandardScaler" else MinMaxScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # ----------------------------- Heatmap
    with st.sidebar.expander("📊 EDA Heatmap"):
        if st.button("Show Heatmap"):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            st.pyplot()

    # ----------------------------- Custom Visualization
    with st.expander("📈 Custom Visualization"):
        st.markdown("Select variables and chart type to visualize your data")

        plot_type = st.selectbox("Choose Plot Type", [
            "Scatter Plot", "Line Plot", "Box Plot", "Histogram", "Bar Plot", "Violin Plot", "Pairplot", "Count Plot"
        ])

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        all_cols = df.columns.tolist()

        if plot_type == "Pairplot":
            pairplot_cols = st.multiselect("Select columns for Pairplot", numeric_cols, default=numeric_cols[:3])
            if st.button("Generate Pairplot"):
                if len(pairplot_cols) >= 2:
                    pairplot_fig = sns.pairplot(df[pairplot_cols])
                    st.pyplot(pairplot_fig)
                else:
                    st.warning("Please select at least 2 numeric columns for Pairplot.")
        else:
            x_axis = st.selectbox("X-axis", all_cols)
            y_axis = st.selectbox("Y-axis", numeric_cols + categorical_cols)
            hue_col = st.selectbox("Hue (Optional - for grouping)", [None] + all_cols)

            if st.button("Generate Plot"):
                try:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    if plot_type == "Scatter Plot":
                        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue_col if hue_col else None, ax=ax)
                    elif plot_type == "Line Plot":
                        sns.lineplot(data=df, x=x_axis, y=y_axis, hue=hue_col if hue_col else None, ax=ax)
                    elif plot_type == "Box Plot":
                        sns.boxplot(data=df, x=x_axis, y=y_axis, hue=hue_col if hue_col else None, ax=ax)
                    elif plot_type == "Histogram":
                        sns.histplot(data=df, x=x_axis, hue=hue_col if hue_col else None, kde=True, bins=30, ax=ax)
                    elif plot_type == "Bar Plot":
                        sns.barplot(data=df, x=x_axis, y=y_axis, hue=hue_col if hue_col else None, ax=ax)
                    elif plot_type == "Violin Plot":
                        sns.violinplot(data=df, x=x_axis, y=y_axis, hue=hue_col if hue_col else None, ax=ax)
                    elif plot_type == "Count Plot":
                        sns.countplot(data=df, x=x_axis, hue=hue_col if hue_col else None, ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating plot: {e}")

    # ----------------------------- ML Section
    with st.expander("🤖 Machine Learning"):
        target = st.selectbox("Select Target Column", df.columns)
        feature_cols = st.multiselect("Select Features", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        features = df[feature_cols]
        X = features.select_dtypes(include=[np.number])
        y = df[target]

        task_type = 'regression' if y.dtype in [np.float64, np.int64] else 'classification'

        model = None
        if task_type == 'classification':
            st.subheader("🧐 Detected as Classification Task")
            model_option = st.selectbox("Choose Classifier", ["RandomForestClassifier", "LogisticRegression", "DecisionTreeClassifier"])
            model = RandomForestClassifier() if model_option == "RandomForestClassifier" else LogisticRegression(max_iter=1000) if model_option == "LogisticRegression" else DecisionTreeClassifier()
        else:
            st.subheader("📈 Detected as Regression Task")
            model_option = st.selectbox("Choose Regressor", ["RandomForestRegressor", "LinearRegression", "DecisionTreeRegressor"])
            model = RandomForestRegressor() if model_option == "RandomForestRegressor" else LinearRegression() if model_option == "LinearRegression" else DecisionTreeRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        with st.container():
            st.markdown("### 🔍 Model Evaluation")
            col1, col2 = st.columns([1, 2])
            with col1:
                if task_type == 'classification':
                    st.text("Classification Report")
                    st.text(classification_report(y_test, y_pred))
                else:
                    st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
                    st.metric("R2 Score", f"{r2_score(y_test, y_pred):.2f}")
            with col2:
                fig, ax = plt.subplots()
                if task_type == 'classification':
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
                    ax.set_title("Confusion Matrix")
                else:
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

        if st.button("Download Trained Model"):
            joblib.dump(model, "trained_model.pkl")
            with open("trained_model.pkl", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:file/output_model.pkl;base64,{b64}" download="trained_model.pkl">Click here to download trained model</a>'
                st.markdown(href, unsafe_allow_html=True)

        if st.checkbox("Run Grid Search (RandomForest only)") and 'RandomForest' in model_option:
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)
            st.write("Best Params:", gs.best_params_)
            st.write("Best Score:", gs.best_score_)

    st.subheader("✅ Cleaned Data Preview")
    st.dataframe(df.head())

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cleaned Data", csv, "cleaned_data.csv", "text/csv")

    # ----------------------------- AI Assistant
    with st.sidebar.expander("💬 AI Assistant", expanded=True):
        st.markdown("Ask me anything about **data cleaning**, **ML**, or **preprocessing**!")

        llm = ChatGroq(
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile"
        )

        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = ConversationBufferMemory()
            st.session_state.chatbot = ConversationChain(llm=llm, memory=st.session_state.chat_memory)

        user_input = st.text_input("You:", key="user_question")

        ml_keywords = [
            "machine learning", "model", "classification", "regression", "random forest",
            "logistic", "linear", "training", "testing", "feature", "target", "scaling",
            "encoding", "imputation", "data cleaning", "transform", "outlier", "heatmap",
            "accuracy", "mse", "r2", "cross-validation", "grid search", "label encoding"
        ]

        def is_ml_question(text):
            return any(keyword in text.lower() for keyword in ml_keywords)

        if user_input:
            if is_ml_question(user_input):
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.chatbot.run(user_input)
            else:
                response = "❗️ I'm designed to assist with machine learning, data cleaning, and preprocessing. Please ask related questions."
            st.markdown(f"**AI:** {response}")
