import pandas as pd

class SeranggaClassifier:
    def __init__(self, all_models):
        self.models = all_models

    def predict(self, X: pd.DataFrame):
        X = X.copy()

        # Level 1: RF untuk semua data
        rf_pred = self.models["rf"].predict(X)

        # Siapkan output akhir (urutannya sama!)
        final_pred = pd.Series(rf_pred, index=X.index, dtype=object)

        # ❗ KELAS YANG TIDAK DIPERCAYA RF
        need_refine = pd.Series(rf_pred).isin(
            ["Aedes_female", "Quinx_female"]
        ).values

        # Level 2: hanya untuk Aedes & Quinx
        if need_refine.any():
            final_pred[need_refine] = self.models["svc"].predict(
                X.loc[need_refine]
            )

        return final_pred.tolist()
