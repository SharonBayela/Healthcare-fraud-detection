"""Provider-grouped splitting, feature engineering and preprocessing."""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

default_seed = 42


def split_by_provider(data, provider_is_fraud, test_size=0.20, val_size=0.20, random_state=default_seed):
    """Split the claims into train, validation and test sets by provider.

    All claims of a provider end up in the same split, and the split keeps the share of fraudulent
    providers similar in each part. Returns the three claim tables and the three provider-id sets.
    """
    provider_table = provider_is_fraud.reset_index()

    train_val_providers, test_providers = train_test_split(
        provider_table, test_size=test_size,
        stratify=provider_table["provider_fraud"], random_state=random_state)
    train_providers, val_providers = train_test_split(
        train_val_providers, test_size=val_size / (1 - test_size),
        stratify=train_val_providers["provider_fraud"], random_state=random_state)

    train_ids = set(train_providers["provider_id"])
    val_ids = set(val_providers["provider_id"])
    test_ids = set(test_providers["provider_id"])
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids), \
        "A provider appears in more than one split"

    train_df = data[data["provider_id"].isin(train_ids)].reset_index(drop=True)
    val_df = data[data["provider_id"].isin(val_ids)].reset_index(drop=True)
    test_df = data[data["provider_id"].isin(test_ids)].reset_index(drop=True)
    return (train_df, val_df, test_df), (train_ids, val_ids, test_ids)


def count_chronic_conditions(column):
    """Turn the chronic_conditions column into a number per patient.
    Works whether the column already holds counts or a text list such as 'diabetes|asthma'."""
    as_numbers = pd.to_numeric(column, errors="coerce")
    if as_numbers.notna().mean() > 0.9:
        return as_numbers

    def count_one(value):
        if pd.isna(value):
            return np.nan
        items = [item.strip().lower() for item in re.split(r"[|;,]", str(value)) if item.strip()]
        return len([item for item in items if item not in ("none", "nan", "0")])

    return column.apply(count_one)


class FeatureBuilder:
    """Builds model features from the merged claim table. No fraud label is ever used.

    The feature ideas follow the paper (age, deceased flag, repeated-claim counts, unique codes per
    provider, provider totals) as far as this dataset allows. It has no admission dates, physicians or
    ICD codes, so those features cannot be reproduced.

    fit()       learns what a normal billed amount looks like for each procedure and severity,
                using TRAIN claims only.
    transform() creates the features for any split. Provider summaries only use that provider's own
                claims, and a provider never sits in two splits, so nothing leaks.
    """

    def fit(self, train_claims):
        log_billed = np.log1p(train_claims["billed_amount"])
        grouped = train_claims.assign(log_billed=log_billed).groupby(["procedure_category", "severity"])["log_billed"]
        self.group_stats = grouped.agg(["mean", "std"])
        self.group_median = train_claims.groupby(["procedure_category", "severity"])["billed_amount"].median()
        self.overall_mean = log_billed.mean()
        self.overall_std = log_billed.std()
        self.overall_median = train_claims["billed_amount"].median()
        return self

    def transform(self, claims_df):
        features = self.claim_features(claims_df)
        features = features.join(self.patient_features(claims_df))
        features = features.join(self.provider_attribute_features(claims_df))
        features = features.join(self.repeat_count_features(claims_df))
        features = features.join(self.provider_summary_features(claims_df, features))
        return features

    def claim_features(self, claims_df):
        """Billing amount, how unusual it is compared with similar claims, and date parts."""
        group_keys = pd.MultiIndex.from_frame(claims_df[["procedure_category", "severity"]])
        stats = self.group_stats.reindex(group_keys)
        typical_mean = stats["mean"].fillna(self.overall_mean).values
        typical_std = stats["std"].fillna(self.overall_std).replace(0, self.overall_std).values
        typical_median = self.group_median.reindex(group_keys).fillna(self.overall_median).values
        log_billed = np.log1p(claims_df["billed_amount"].values)

        out = pd.DataFrame(index=claims_df.index)
        out["severity"] = claims_df["severity"].values
        out["billed_amount"] = claims_df["billed_amount"].values
        out["log_billed"] = log_billed
        out["peer_z"] = (log_billed - typical_mean) / typical_std
        out["peer_ratio"] = claims_df["billed_amount"].values / np.maximum(typical_median, 1e-6)
        out["procedure_category"] = claims_df["procedure_category"].values
        out["diagnosis_group"] = claims_df["diagnosis_group"].values
        out["month"] = claims_df["service_date"].dt.month.values
        out["day_of_week"] = claims_df["service_date"].dt.dayofweek.values
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
        return out

    def patient_features(self, claims_df):
        """Age, sex, region, chronic condition count, deceased flag and billing after death."""
        out = pd.DataFrame(index=claims_df.index)
        if "age" in claims_df:
            out["age"] = pd.to_numeric(claims_df["age"], errors="coerce").values
        if "sex" in claims_df:
            out["sex"] = claims_df["sex"].values
        if "patient_region" in claims_df:
            out["patient_region"] = claims_df["patient_region"].values
        if "chronic_conditions" in claims_df:
            out["n_chronic"] = count_chronic_conditions(claims_df["chronic_conditions"]).values
        if "death_date" in claims_df:
            out["patient_deceased"] = claims_df["death_date"].notna().astype(int).values
            billed_after_death = claims_df["service_date"] > claims_df["death_date"]
            out["billed_after_death"] = billed_after_death.fillna(False).astype(int).values
        return out

    def provider_attribute_features(self, claims_df):
        """Provider specialty, region, size and years active (whichever exist)."""
        out = pd.DataFrame(index=claims_df.index)
        for column in ["specialty", "provider_region", "provider_size", "years_active"]:
            if column in claims_df:
                out[column] = claims_df[column].values
        return out

    def repeat_count_features(self, claims_df):
        """How often the same combination shows up (the paper's repeated-claim counts)."""
        out = pd.DataFrame(index=claims_df.index)

        same_visit = claims_df.groupby(["provider_id", "patient_id", "service_date", "procedure_category"])["claim_id"].transform("size")
        out["same_visit_count"] = same_visit.values
        out["is_duplicate"] = (same_visit > 1).astype(int).values

        combinations = {
            "count_provider_patient": ["provider_id", "patient_id"],
            "count_provider_patient_procedure": ["provider_id", "patient_id", "procedure_category"],
            "count_provider_patient_diagnosis": ["provider_id", "patient_id", "diagnosis_group"],
            "count_provider_procedure": ["provider_id", "procedure_category"],
            "count_provider_diagnosis": ["provider_id", "diagnosis_group"],
        }
        for new_name, group_columns in combinations.items():
            out[new_name] = claims_df.groupby(group_columns)["claim_id"].transform("size").values
        return out

    def provider_summary_features(self, claims_df, features_so_far):
        """One summary per provider (totals, averages, unique codes), copied onto each of its claims."""
        summary_input = pd.DataFrame({
            "provider_id": claims_df["provider_id"].values,
            "patient_id": claims_df["patient_id"].values,
            "procedure": claims_df["procedure_category"].values,
            "diagnosis": claims_df["diagnosis_group"].values,
            "severity": claims_df["severity"].values,
            "billed": claims_df["billed_amount"].values,
            "peer_z": features_so_far["peer_z"].values,
            "is_duplicate": features_so_far["is_duplicate"].values,
            "high_severity": (claims_df["severity"].values >= 4).astype(int),
        })
        if "billed_after_death" in features_so_far:
            summary_input["after_death"] = features_so_far["billed_after_death"].values
        else:
            summary_input["after_death"] = 0

        summary = summary_input.groupby("provider_id").agg(
            provider_n_claims=("patient_id", "size"),
            provider_n_patients=("patient_id", "nunique"),
            provider_n_procedures=("procedure", "nunique"),
            provider_n_diagnoses=("diagnosis", "nunique"),
            provider_total_billed=("billed", "sum"),
            provider_mean_billed=("billed", "mean"),
            provider_mean_severity=("severity", "mean"),
            provider_share_high_severity=("high_severity", "mean"),
            provider_mean_peer_z=("peer_z", "mean"),
            provider_duplicate_rate=("is_duplicate", "mean"),
            provider_after_death_rate=("after_death", "mean"))
        summary["provider_claims_per_patient"] = summary["provider_n_claims"] / summary["provider_n_patients"]

        per_claim = summary.reindex(claims_df["provider_id"].values)
        per_claim.index = claims_df.index
        return per_claim


def build_feature_matrices(train_df, val_df, test_df):
    """Build features for the three splits, drop constant columns, then impute, scale and one-hot encode.

    Everything is fitted on the training data only. Returns a dict with x_train, x_val, x_test,
    feature_names, numeric_cols, category_cols, constant_cols, the feature tables and the fitted objects.
    """
    feature_builder = FeatureBuilder().fit(train_df)
    features_train = feature_builder.transform(train_df)
    features_val = feature_builder.transform(val_df)
    features_test = feature_builder.transform(test_df)

    numeric_cols = features_train.select_dtypes(include="number").columns.tolist()
    constant_cols = [c for c in numeric_cols if features_train[c].nunique() <= 1]
    features_train = features_train.drop(columns=constant_cols)
    features_val = features_val.drop(columns=constant_cols)
    features_test = features_test.drop(columns=constant_cols)
    numeric_cols = [c for c in numeric_cols if c not in constant_cols]

    category_cols = [c for c in features_train.columns if c not in numeric_cols]
    for feature_table in (features_train, features_val, features_test):
        for column in category_cols:
            feature_table[column] = feature_table[column].astype(str)

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", RobustScaler())]), numeric_cols),
        ("category", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False), category_cols),
    ])
    x_train = preprocessor.fit_transform(features_train).astype(np.float32)
    x_val = preprocessor.transform(features_val).astype(np.float32)
    x_test = preprocessor.transform(features_test).astype(np.float32)
    feature_names = np.array([name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()])

    return {"x_train": x_train, "x_val": x_val, "x_test": x_test, "feature_names": feature_names,
            "numeric_cols": numeric_cols, "category_cols": category_cols, "constant_cols": constant_cols,
            "features_train": features_train, "features_val": features_val, "features_test": features_test,
            "feature_builder": feature_builder, "preprocessor": preprocessor}
