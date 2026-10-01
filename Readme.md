# Project Structure
    .
    ├─ expriments/                    # 
        ├─ exp_basic.py               # exp model setting
        ├─ ...

    ├─ layers/                        #
        ├─ Embed.py                   # Embedding
        ├─ Invertible.py              # ReInv for Rlinear
        ├─ selfAttention_Family.py    # Different  attention for Tansformer
        |─ Transformer_EncDec.py      # Transformer encoder and decoder
        ├─ Taylorder.py               # Talyor order for DAMIE
        ├─ TimeSpaceFlip.py           # T2V-Q in attention
        ├─ TayloTrans_Enc.py          # Encoder for DAMIE

    ├─ model/                         
        ├─ RLinear.py                 # RLinear model in LN
        ├─ SOFTS.py                   # SoFTS model in ND
        ├─ iTransformer.py            # iTransformer model in ND
        ├─ DAMIE_TaylorTrans.py       # Our DAMIE model 

    ├─ script/
        ├─ DAMIE/                     # DAMIE exp configs in six dataset
            ├─ ETT/ 
                ├─ ETTh1.sh                
                ├─ ETTh2.sh                
                ├─ ETTm1.sh                
                ├─ ETTm2.sh                
            ├─ TMT_weather.sh             
            ├─ TMT_weather.sh            
        ├─ ... 
    ├─ ... 

    └─ run.py                         # Args in experiment 





## 🌟 Getting Start

### 🛠️ Installation

```bash
pip install -r requirements.txt
```

### 📦 Datasets

The datasets can be obtained from [here](https://github.com/wzhwzhwzh0921/S-D-Mamba/releases/download/datasets/S-Mamba_datasets.zip).

### 🚀 Train and evaluate

```bash
# ETT
bash ./scripts/DAMIE/ETT/ETTh1.sh
bash ./scripts/DAMIE/ETT/ETTh2.sh
bash ./scripts/DAMIE/ETT/ETTm1.sh
bash ./scripts/DAMIE/ETT/ETTm2.sh

# Weather
bash ./scripts/DAMIE/TMT_weather.sh

# ECL
bash ./scripts/DAMIE/ECL.sh
```
