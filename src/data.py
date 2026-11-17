"""Loading, labelling and merging the three claim tables."""
import pandas as pd


def load_tables(repo_id="quailrobot/health-claims-v1-sample"):
    """Read the claims, patients and providers csv files from the Hugging Face dataset repo."""
    from huggingface_hub import HfApi

    all_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    csv_files = [f for f in all_files if f.lower().endswith(".csv")]
    tables = {}
    for table_name in ["claims", "patients", "providers"]:
        matches = [f for f in csv_files if table_name in f.lower()]
        assert matches, f"No csv file containing '{table_name}' was found. Files in the repo: {all_files}"
        tables[table_name] = pd.read_csv(f"hf://datasets/{repo_id}/{matches[0]}")
    return tables["claims"], tables["patients"], tables["providers"]


def make_provider_labels(claims, providers):
    """Provider-level fraud label (used only for splitting and plots, never as a model input).

    A provider counts as fraudulent if it has at least one fraudulent claim, unless the providers
    table already contains its own provider-level label.
    """
    label_from_claims = claims.groupby("provider_id")["is_fraud"].max()
    has_provider_labels = ("is_fraud_provider" in providers.columns
                           and providers["is_fraud_provider"].notna().any())
    if has_provider_labels:
        labels = providers.drop_duplicates("provider_id").set_index("provider_id")["is_fraud_provider"]
        labels = labels.reindex(label_from_claims.index).fillna(label_from_claims)
    else:
        labels = label_from_claims
    return labels.astype(int).rename("provider_fraud")


def merge_tables(claims, patients, providers):
    """Merge claims + patients + providers.

    The provider label columns are dropped first so they cannot leak into the features.
    """
    label_columns = [c for c in ["is_fraud_provider", "fraud_scheme"] if c in providers.columns]
    provider_info = (providers.drop(columns=label_columns)
                              .drop_duplicates("provider_id")
                              .rename(columns={"region": "provider_region", "size": "provider_size"}))
    patient_info = (patients.drop_duplicates("patient_id")
                            .rename(columns={"region": "patient_region"}))
    return (claims.merge(patient_info, on="patient_id", how="left")
                  .merge(provider_info, on="provider_id", how="left"))


def prepare_dates(data):
    """Convert date columns and sort the claims by service date."""
    data = data.copy()
    data["service_date"] = pd.to_datetime(data["service_date"], errors="coerce")
    if "death_date" in data.columns:
        data["death_date"] = pd.to_datetime(data["death_date"], errors="coerce")
    return data.sort_values("service_date").reset_index(drop=True)


def load_and_merge(repo_id="quailrobot/health-claims-v1-sample"):
    """Convenience wrapper: returns (data, provider_is_fraud)."""
    claims, patients, providers = load_tables(repo_id)
    provider_is_fraud = make_provider_labels(claims, providers)
    data = prepare_dates(merge_tables(claims, patients, providers))
    return data, provider_is_fraud
