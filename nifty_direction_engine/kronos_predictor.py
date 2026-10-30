import torch
import pandas as pd
import sys
from pathlib import Path

# Path adjustments
CUR_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CUR_DIR / "kronos_lib"))

from model import Kronos, KronosTokenizer, KronosPredictor

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

class KronosDirectionPredictor:
    def __init__(self, model_name=KRONOS_MODEL_NAME, tokenizer_name=TOKENIZER_NAME, device=None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        self.model = Kronos.from_pretrained(model_name)
        self.predictor = KronosPredictor(self.model, self.tokenizer, device=self.device, max_context=512)
        
    def get_prediction(self, x_df: pd.DataFrame, x_timestamp: pd.Series = None, pred_len: int = 18):
        """
        Generates forecast for next 18 steps (10:00 AM -> 11:30 AM).
        """
        if x_timestamp is None:
            last_time = pd.Timestamp.now()
            x_timestamp = pd.Series([last_time - pd.Timedelta(minutes=5 * (len(x_df) - i)) for i in range(len(x_df))])
            
        last_time = x_timestamp.iloc[-1]
        y_timestamp = pd.Series([last_time + pd.Timedelta(minutes=5 * i) for i in range(1, pred_len + 1)])
        
        pred_df = self.predictor.predict(
            df=x_df[['open', 'high', 'low', 'close']],
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.7,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )
        
        spot_10am = float(x_df['close'].iloc[-1])
        proj_close = float(pred_df['close'].iloc[-1])
        predicted_return = (proj_close - spot_10am) / spot_10am
        
        return predicted_return, pred_df
