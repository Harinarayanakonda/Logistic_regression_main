import pandas as pd

class DataLoader:
    def load_data(self, file_obj):
        # Try to read as CSV, Excel, or JSON based on file_obj type
        # The calling code should ensure the correct file_obj type is passed
        try:
            # Try CSV
            return pd.read_csv(file_obj)
        except Exception:
            try:
                # Try Excel
                file_obj.seek(0)
                return pd.read_excel(file_obj)
            except Exception:
                try:
                    # Try JSON
                    file_obj.seek(0)
                    return pd.read_json(file_obj)
                except Exception as e:
                    raise ValueError(f"Unsupported or corrupted file: {e}")